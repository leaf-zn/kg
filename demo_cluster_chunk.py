import os
import logging
import json
import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from py2neo import Graph, Node, Relationship, NodeMatcher

# 下载NLTK数据（如果还没有）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 设置日志以便调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def store_in_neo4j(graph_data):
    """
    将知识图谱存储到Neo4j数据库

    Args:
        graph_data (dict): 包含实体和关系的知识图谱数据
    """
    try:
        # 连接到Neo4j数据库
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = "1234567890"

        logger.info(f"正在连接Neo4j数据库: {neo4j_uri}")
        neo4j_graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # 清空现有数据 (可选)
        logger.info("清空现有数据...")
        neo4j_graph.run("MATCH (n) DETACH DELETE n")

        # 创建一个节点匹配器
        matcher = NodeMatcher(neo4j_graph)

        # 开始一个事务
        tx = neo4j_graph.begin()
        logger.info(f"开始创建节点，实体总数: {len(graph_data['entities'])}")

        # 创建实体节点，避免重复创建
        entity_nodes = {}
        for entity in graph_data["entities"]:
            # 清理实体名称（去除前后空格）
            clean_entity = entity.strip()
            if not clean_entity:
                continue

            # 创建新节点 (不检查是否存在，因为已经清空了数据库)
            node = Node("Entity", name=clean_entity)
            tx.create(node)
            entity_nodes[clean_entity] = node
            logger.info(f"创建实体节点: {clean_entity}")

        # 为实体添加聚类信息
        for cluster_id, entities in graph_data["entity_clusters"].items():
            for entity in entities:
                clean_entity = entity.strip()
                if clean_entity in entity_nodes:
                    entity_nodes[clean_entity]["cluster"] = cluster_id
                    tx.push(entity_nodes[clean_entity])  # 确保属性更新被保存

        # 提交节点创建事务
        neo4j_graph.commit(tx)
        logger.info(f"成功创建 {len(entity_nodes)} 个实体节点")

        # 创建额外的缺失实体节点
        extra_tx = neo4j_graph.begin()
        missing_entities_created = False

        for rel in graph_data["relations"]:
            if len(rel) != 3:
                continue

            source, relation_type, target = rel

            # 清理实体名称
            clean_source = source.strip()
            clean_target = target.strip()

            # 为缺失的实体创建节点
            for entity_name in [clean_source, clean_target]:
                if entity_name and entity_name not in entity_nodes:
                    # 创建新节点
                    node = Node("Entity", name=entity_name)
                    extra_tx.create(node)
                    entity_nodes[entity_name] = node
                    missing_entities_created = True
                    logger.info(f"为关系创建缺失实体节点: {entity_name}")

        # 提交额外节点创建事务（如果有实体被创建）
        if missing_entities_created:
            neo4j_graph.commit(extra_tx)

        # 打印所有已创建的节点用于调试
        logger.info("已创建的所有实体节点:")
        for entity_name in entity_nodes:
            logger.info(f"  - {entity_name}")

        # 创建关系 - 使用单独的事务
        logger.info(f"开始创建关系，关系总数: {len(graph_data['relations'])}")
        relations_count = 0

        # 新建一个事务用于关系创建
        rel_tx = neo4j_graph.begin()

        for rel in graph_data["relations"]:
            if len(rel) != 3:
                logger.warning(f"关系数据格式不正确，跳过: {rel}")
                continue

            source, relation_type, target = rel

            # 清理实体名称和关系类型
            clean_source = source.strip()
            clean_relation = relation_type.strip()
            clean_target = target.strip()

            # 标准化关系名称 (移除特殊字符，保留字母、数字和下划线)
            normalized_relation = re.sub(r'[^\w]', '_', clean_relation)
            if not normalized_relation:
                normalized_relation = "RELATED_TO"

            # 验证源节点和目标节点
            logger.info(f"尝试创建关系: '{clean_source}' --[{normalized_relation}]--> '{clean_target}'")

            # 使用节点字典查找节点
            source_node = entity_nodes.get(clean_source)
            target_node = entity_nodes.get(clean_target)

            if source_node and target_node:
                # 在事务内创建关系
                relationship = Relationship(source_node, normalized_relation, target_node)
                rel_tx.create(relationship)  # 使用create而不是merge
                relations_count += 1
                logger.info(f"成功创建关系: {clean_source} --[{normalized_relation}]--> {clean_target}")
            else:
                if not source_node:
                    logger.warning(f"源实体节点不存在: '{clean_source}'")
                if not target_node:
                    logger.warning(f"目标实体节点不存在: '{clean_target}'")

        # 提交关系创建事务
        neo4j_graph.commit(rel_tx)
        logger.info(f"成功创建 {relations_count} 个关系")

        # 打印示例Cypher查询，以便用户在Neo4j浏览器中查看图谱
        print("\n要在Neo4j浏览器中查看知识图谱，请运行以下查询:")
        print("MATCH (n:Entity) RETURN n LIMIT 100;")
        print("MATCH (n:Entity)-[r]->(m:Entity) RETURN n, r, m LIMIT 100;")

        # 通过直接查询验证关系是否创建成功
        verify_result = neo4j_graph.run("MATCH ()-[r]->() RETURN count(r) AS rel_count").data()
        rel_count = verify_result[0]['rel_count'] if verify_result else 0
        logger.info(f"Neo4j数据库中验证的关系数量: {rel_count}")
   
        return True

    except Exception as e:
        logger.error(f"Neo4j数据库操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def chunk_text(text, chunk_size=5):
    """
    将文本分成多个块，每个块包含指定数量的句子

    Args:
        text (str): 输入文本
        chunk_size (int): 每个块中的句子数

    Returns:
        list: 文本块列表
    """
    # 分割句子
    sentences = sent_tokenize(text)
    logger.info(f"文本共分割为 {len(sentences)} 个句子")

    # 创建块
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i + chunk_size])
        chunks.append(chunk)

    logger.info(f"文本分割为 {len(chunks)} 个块")
    return chunks


def extract_entities_from_chunk(base_url, model_name, chunk):
    """从文本块中提取实体"""
    # entities_prompt = f"""
    # Task: Extract all important entities (people, places, organizations, concepts) from the following text.
    #
    # DO NOT translate entity names. Keep them in their original language (English).
    #
    # Text:
    # {chunk}
    #
    # Please return only a list of entities in JSON array format: ["entity1", "entity2", ...]
    # """

    entities_prompt = f"""
        Task: Extract all important entities from the following text related to Military aviation and tactical aviation operations.

        Focus on extracting these specific types of entities:
        1. Aircraft types (e.g., F-16, Su-30, B-2, etc.)
        2. Weapon Systems (e.g., AIM-120, AGM-88, etc.)
        3. Personnel roles (e.g., pilot, navigator, etc.)
        4. Tactics and techniques (e.g., dogfighting, bombing, etc.)
        5. Military units (e.g., 56th Fighter Wing, 7th Bomb Wing, etc.)
        6. Sensors and electronic systems (e.g., radar, EW systems, etc.)
        7. Performance Parameters (e.g., speed, range, payload, etc.)
        8. Other entities related to Military aviation and tactical aviation operations.
        
        DO NOT translate entity names. Keep them in their original language.

        Text:
        {chunk}

        Please return only a list of entities in JSON array format: ["entity1", "entity2", ...]
        """

    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "prompt": entities_prompt,
            "stream": False
        }
    )

    if response.status_code != 200:
        logger.error(f"API调用失败: {response.status_code} - {response.text}")
        return set()

    entities_content = response.json().get("response", "")

    # 尝试从响应中提取JSON数组
    json_match = re.search(r'\[.*\]', entities_content, re.DOTALL)
    if json_match:
        entities_json = json_match.group(0)
        entities_json = entities_json.replace("'", '"').replace('\n', '')
        try:
            chunk_entities = set(json.loads(entities_json))
        except json.JSONDecodeError:
            entity_matches = re.findall(r'"([^"]+)"', entities_json)
            chunk_entities = set(entity_matches)
    else:
        entity_matches = re.findall(r'"([^"]+)"', entities_content)
        if not entity_matches:
            entity_matches = re.findall(r'[\w\s]+', entities_content)
        chunk_entities = set(entity_matches)

    # 过滤空实体
    chunk_entities = {e.strip() for e in chunk_entities if e.strip()}
    return chunk_entities


def cluster_entities(entities, n_clusters=3):
    """
    将实体聚类分组

    Args:
        entities (list): 实体列表
        n_clusters (int): 聚类数量

    Returns:
        dict: 聚类结果，键为簇ID，值为该簇中的实体列表
    """
    if len(entities) < n_clusters:
        logger.warning(f"实体数量({len(entities)})小于请求的聚类数量({n_clusters})，将调整聚类数量")
        n_clusters = max(1, len(entities) // 2)

    if len(entities) <= 1:
        return {0: list(entities)}

    # 使用TF-IDF向量化实体
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform([str(entity) for entity in entities])

        # 如果实体太少，返回一个簇
        if X.shape[0] < n_clusters:
            return {0: list(entities)}

        # 执行KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # 组织聚类结果
        clusters = {}
        for i, entity in enumerate(entities):
            cluster_id = kmeans.labels_[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(entity)

        return clusters
    except Exception as e:
        logger.error(f"聚类过程中出错: {e}")
        return {0: list(entities)}


def main():
    # 使用Ollama地址
    base_url = os.getenv("BASE_URL", "http://localhost:11434")
    logger.info(f"使用Ollama API基地址: {base_url}")

    # 测试API连接
    try:
        logger.info("测试API连接...")
        response = requests.get(f"{base_url}/api/version")
        logger.info(f"API连接测试结果: {response.status_code}")
        logger.info(f"API版本: {response.json()}")
    except Exception as e:
        logger.error(f"连接测试失败: {e}")
        return

    # 获取可用模型
    try:
        logger.info("获取可用模型...")
        response = requests.get(f"{base_url}/api/tags")
        models = response.json().get("models", [])
        logger.info(f"可用模型: {models}")

        # 确认模型是否可用
        model_name = "qwen2.5:latest"
        if not any(model.get("name") == model_name for model in models):
            logger.warning(f"未找到模型 {model_name}，将使用第一个可用模型")
            if models:
                model_name = models[0].get("name", "qwen2.5:latest")
    except Exception as e:
        logger.warning(f"获取模型列表失败: {e}，将使用默认模型: qwen2.5:latest")
        model_name = "qwen2.5:latest"

    logger.info(f"使用模型: {model_name}")

    # 输入文本 - 可以是更长的文本
    # text_input = """
    # Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father.
    # Josh is a student at XYZ University. He studies computer science.
    # Andrew works as a software engineer at ABC Corporation.
    # Linda is a teacher at a local high school.
    # The XYZ University has a renowned Computer Science department.
    # Computer Science is a field that studies computation and information processing.
    # ABC Corporation is a leading technology company in the software industry.
    # The software industry creates and maintains software applications and systems.
    # High schools provide education for students typically between the ages of 14 and 18.
    # """

    text_input = """  

背景
2025年4月15日，北约与某区域势力在波罗的海争议空域爆发冲突.美国空军第48战斗机联队的2架F-35A“闪电II”（编号AF-21/22）奉命拦截俄罗斯空天军第4航空团的4架Su-57“重罪犯”（编号RF-701~704）.此次对抗涉及隐身战机、数据链协同与电子战系统交锋.

任务阶段
预警与部署

北约E-7A“楔尾”预警机（呼号EYES-1）通过AN/APY-2雷达在250公里外发现Su-57编队，并利用MADL数据链向F-35A传输目标参数.
Su-57编队启动N036 Byelka雷达的“低截获概率”（LPI）模式，同时释放L-402“希比内”电子对抗吊舱干扰GPS信号.
超视距攻击

F-35A飞行员约翰·米勒少校（代号Falcon-1）在150公里距离发射AIM-120D导弹，依赖机载AN/ASQ-239 Barracuda系统维持对RF-701的锁定.
Su-57编队迅速散开为双机战术组（RF-701/702主攻，RF-703/704掩护），发射R-37M远程导弹反击，并启用红外对抗系统（IRCM）干扰红外制导.
电子战博弈

F-35A开启AN/ALQ-214干扰机压制Su-57的雷达回波，导致RF-702的N036雷达丢失目标3秒.
Su-703启动“希比内”吊舱的主动干扰模式，成功诱偏一枚AIM-120D导弹至预设的电子欺骗区域.
近距机动与脱离

当双方距离缩短至30公里时，Falcon-1执行“高G滚筒”机动躲避R-37M，同时释放ALE-70诱饵弹干扰敌方导弹导引头.
RF-704因燃油不足（原计划由伊尔-78加油机补给未到位）被迫脱离战场，剩余Su-57编队向加里宁格勒基地撤退.
结果与影响
本次对抗未造成击落，但暴露Su-57的LPI雷达在对抗F-35A的APG-81雷达时存在多普勒分辨率劣势.
北约事后将对抗数据输入“红旗军演-26”仿真系统，用于优化隐身战机与预警机的协同战术.
俄罗斯宣布加快Kh-74M2超远程空空导弹的列装进度，以应对AIM-260的威胁.

在现代军事航空领域，F-35 Lightning II多用途战斗机因其卓越的隐身性能和多功能性成为许多国家空军的核心装备.该机型配备了AN/APG-81 AESA雷达，能够同时执行空对空和空对地任务，其最大速度可达1.6马赫，实用升限超过50,000英尺.在武器系统方面，F-35内置一门GAU-22/A 25毫米机炮，并可携带多种制导武器，包括AIM-120 AMRAAM中程空空导弹和JDAM联合直接攻击弹药.此外，F-35还配备了AN/ASQ-239电子战系统，能够对敌方雷达进行干扰和压制.
在战术执行中，F-35常与Su-35 Flanker-E战斗机进行对抗演练.Su-35以其强大的推力矢量控制（TVC）系统和Irbis-E无源相控阵雷达著称，能够执行复杂的近距离格斗战术.Su-35的最大速度为2.25马赫，并可挂载R-77中程空空导弹和Kh-38ME空地导弹.此外，Su-35还配备了OLS-35红外搜索与跟踪系统（IRST），能够在雷达静默状态下探测敌方目标.
在联合战术行动中，Eurofighter Typhoon战斗机常作为防空拦截力量参与作战.该机型配备ECR.90雷达，最大速度为2.0马赫，并可携带ASRAAM近程空空导弹和流星远程空空导弹.Typhoon的武器系统操作员和雷达拦截官需密切协作，以确保在**超视距空战（BVR）中占据优势.此外，Typhoon还具备地形跟踪飞行（TF）**能力，能够在低空突防任务中规避敌方雷达探测.
在电子战领域，EA-18G Growler电子战机发挥着关键作用.该机型搭载AN/ALQ-218电子支援系统和AN/ALQ-99电子对抗系统，能够对敌方雷达进行雷达干扰和通信压制.在执行任务时，电子战军官（EWO）负责操控这些系统，确保己方战机免受敌方防空系统的威胁.此外，EA-18G还配备了AN/ALQ-249先进电子攻击系统，能够执行更复杂的电子战任务.
在战术层面，现代战斗机常采用**“双四”编队**（即两架战机组成四机编队）执行任务.这种编队结合了**“长机-僚机”协同作战和“高低搭配”战术**，能够在复杂战场环境中实现灵活的战术部署.此外，**地形跟踪飞行（TF）技术被广泛应用于低空突防任务，以规避敌方雷达探测.在执行任务时，飞行员需密切配合，利用头盔瞄准具（HMD）和数据链系统（Link 16）**实现高效的目标识别和信息共享.
在性能参数方面，F-22 Raptor战斗机以其2.25马赫的最大速度和超过38,000英尺的升限成为隐身战斗机的标杆.其AN/APG-77 AESA雷达具备强大的目标探测和跟踪能力，而推力矢量控制（TVC）系统使其在近距离格斗中具备卓越的机动性.此外，F-22还配备了AN/ALE-52诱饵弹发射系统，能够有效对抗敌方导弹攻击.
在现代空战中，F-15EX Eagle II战斗机作为一款先进的多用途战机，能够执行复杂的空对空和空对地任务.该机型配备了AN/APG-82(V)1有源相控阵雷达，最大速度为2.5马赫，并可携带AIM-9X近程空空导弹和GBU-31联合直接攻击弹药.此外，F-15EX还具备**“保形油箱（CFT）”技术**，显著提高了燃油效率和作战半径.
在防空压制任务中，F/A-18E/F Super Hornet战斗机常被用于执行SEAD（抑制作战防空）任务.该机型配备了AN/APG-79 AESA雷达，并可携带AGM-88 HARM反辐射导弹，用于摧毁敌方雷达和防空系统.此外，Super Hornet还具备空中加油能力，能够延长任务时间并扩大作战范围.
在战术训练中，F-16 Fighting Falcon战斗机因其灵活性和多功能性被广泛用于各种作战场景.该机型配备了AN/APG-83可变参数雷达，最大速度为2.0马赫，并可携带AIM-120 AMRAAM和GBU-12激光制导炸弹.F-16还具备头盔瞄准具（HMD），使飞行员能够在复杂环境中快速锁定目标.
在现代电子战中，Su-34 Fullback战斗轰炸机以其强大的对地攻击能力和电子战系统著称.该机型配备了N012“雪豹”雷达，最大速度为1.8马赫，并可携带Kh-59MK巡航导弹和KAB-500激光制导炸弹.此外，Su-34还配备了L175M“希比”雷达干扰系统，能够有效对抗敌方雷达和导弹攻击.
在现代军事航空领域，JF-17 Thunder战斗机因其出色的性价比和多功能性成为许多国家空军的重要装备.该机型配备了KLJ-7 AESA雷达，能够执行空对空和空对地任务，其最大速度可达1.6马赫，实用升限超过17,000米.在武器系统方面，JF-17可携带PL-12中程空空导弹和CM-400AKG超音速空地导弹，并内置一门23毫米双管航炮.
在对抗演练中，JF-17常与J-20 Mighty Dragon隐身战斗机协同作战.J-20以其卓越的隐身性能和先进的光电分布式孔径系统（EODAS）著称，能够执行复杂的超视距空战（BVR）任务.J-20的最大速度为2.0马赫，并配备AN/APG-83 AESA雷达，可携带PL-15远程空空导弹和YJ-91反辐射导弹.
在电子战领域，Su-30SM战斗机因其强大的雷达干扰能力和Kh-58UShKE反辐射导弹而备受关注.该机型配备了N011M“贝克尔”雷达，最大速度为2.2马赫，并具备**地形跟踪飞行（TF）**能力，能够在复杂地形中执行低空突防任务.
在联合战术行动中，F/A-18F Super Hornet常作为多用途战机参与作战.该机型配备AN/APG-79 AESA雷达，最大速度为1.8马赫，并可携带AIM-9X近程空空导弹和GBU-54激光制导炸弹.此外，Super Hornet还具备空中加油能力，能够延长任务时间并扩大作战范围.
在防空压制任务中，EA-18G Growler电子战机发挥着关键作用.该机型搭载AN/ALQ-218电子支援系统和AN/ALQ-99电子对抗系统，能够对敌方雷达进行雷达干扰和通信压制.在执行任务时，**电子战军官（EWO）**负责操控这些系统，确保己方战机免受敌方防空系统的威胁.
在战术层面，现代战斗机常采用**“双机编队”执行任务.这种编队结合了“长机-僚机”协同作战和“高低搭配”战术**，能够在复杂战场环境中实现灵活的战术部署.此外，**地形跟踪飞行（TF）**技术被广泛应用于低空突防任务，以规避敌方雷达探测.
在性能参数方面，F-15C Eagle战斗机以其2.5马赫的最大速度和超过20,000米的升限成为防空拦截的标杆.其AN/APG-63(V)3 AESA雷达具备强大的目标探测和跟踪能力，而推力矢量控制（TVC）系统使其在近距离格斗中具备卓越的机动性.
在现代空战中，Su-33 Flanker-D舰载战斗机因其强大的对海攻击能力和Kh-31反舰导弹而备受关注.该机型配备了N019“屏障”雷达，最大速度为2.15马赫，并具备空中加油能力，能够延长任务时间并扩大作战范围.
在战术训练中，MiG-29SMT战斗机因其灵活性和多功能性被广泛用于各种作战场景.该机型配备了N019M“甲虫-M”雷达，最大速度为2.25马赫，并可携带R-73近程空空导弹和KAB-500激光制导炸弹.
在电子战领域，G550 CAEW预警机因其强大的EL/M-2075“蓝斯波特”雷达和**数据链系统（Link 16）而备受关注.该机型能够执行空中预警与控制（AWACS）**任务，并为友军提供实时战场信息和目标指示.
在现代军事航空中，MQ-9 Reaper无人机因其强大的侦察和攻击能力而被广泛使用.该机型配备了AN/APY-8雷达和地狱火导弹，能够执行长时间侦察和精确打击任务.此外，MQ-9还具备卫星通信能力，能够在全球范围内执行任务.
在战术层面，现代战斗机常采用**“四机编队”执行任务.这种编队结合了“双长机”协同作战和“分散攻击”战术**，能够在复杂战场环境中实现灵活的战术部署.此外，**电子战支援（ESM）**技术被广泛应用于对抗敌方雷达和通信系统.
在性能参数方面，F-16C/D Block 50战斗机以其2.0马赫的最大速度和超过15,000米的升限成为多用途战机的代表.其AN/APG-83可变参数雷达具备强大的目标探测和跟踪能力，而**头盔瞄准具（HMD）**使飞行员能够在复杂环境中快速锁定目标.
在现代空战中，J-11B Flanker-B战斗机因其强大的雷达干扰能力和PL-10近程空空导弹而备受关注.该机型配备了KLJ-7A AESA雷达，最大速度为2.25马赫，并具备空中加油能力，能够延长任务时间并扩大作战范围.
在战术训练中，F-5E/F Tiger II战斗机因其灵活性和低成本被广泛用于模拟敌方战机.该机型配备了AN/APQ-159雷达，最大速度为1.6马赫，并可携带AIM-9 Sidewinder近程空空导弹和Mk-82炸弹.
在电子战领域，EC-130H Compass Call电子战飞机因其强大的通信干扰能力和心理战广播系统而备受关注.该机型能够对敌方通信系统进行干扰，并传播心理战信息以削弱敌方士气.
在现代军事航空中，A-10C Thunderbolt II攻击机因其强大的对地攻击能力和GAU-8/A 30毫米航炮而备受关注.该机型配备了AN/APG-68雷达，最大速度为0.75马赫，并可携带AGM-65小牛导弹和Mk-82炸弹.
在战术层面，现代战斗机常采用**“六机编队”执行任务.这种编队结合了“双长机”协同作战和“多层次攻击”战术**，能够在复杂战场环境中实现灵活的战术部署.此外，**数据链系统（Link 16）**被广泛应用于友军之间的信息共享和协同作战.
在性能参数方面，Su-27 Flanker-B战斗机以其2.35马赫的最大速度和超过19,000米的升限成为防空拦截的标杆.其N001雷达具备强大的目标探测和跟踪能力，而推力矢量控制（TVC）系统使其在近距离格斗中具备卓越的机动性.
在现代空战中，F-15E Strike Eagle战斗轰炸机因其强大的对地攻击能力和AN/APG-70 AESA雷达而备受关注.该机型最大速度为2.5马赫，并可携带GBU-28激光制导炸弹和AGM-88 HARM反辐射导弹.
在战术训练中，F-16CJ战斗机因其强大的电子战能力和AN/ALQ-131电子对抗吊舱而被广泛用于执行SEAD（抑制作战防空）任务.该机型配备了AN/APG-68雷达，最大速度为2.0马赫，并可携带AGM-88 HARM反辐射导弹.
在电子战领域，EA-6B Prowler电子战机因其强大的AN/ALQ-99电子对抗系统而备受关注.该机型能够对敌方雷达进行干扰，并为友军提供电子保护.在执行任务时，**电子战军官（EWO）**负责操控这些系统，确保己方战机免受敌方防空系统的威胁.

  """

    # 生成知识图谱
    try:
        logger.info("开始生成知识图谱...")

        # 1. 分块处理
        chunk_size = 3  # 每个块的句子数
        chunks = chunk_text(text_input, chunk_size)

        # 2. 从每个块中提取实体
        all_entities = set()
        for i, chunk in enumerate(chunks):
            logger.info(f"处理块 {i + 1}/{len(chunks)}...")
            chunk_entities = extract_entities_from_chunk(base_url, model_name, chunk)
            logger.info(f"块 {i + 1} 中提取到的实体: {chunk_entities}")
            all_entities.update(chunk_entities)

        logger.info(f"所有块中提取到的实体总数: {len(all_entities)}")

        if all_entities:
            # 3. 实体聚类
            n_clusters = min(3, len(all_entities))  # 设置聚类数量，但不超过实体总数
            entity_clusters = cluster_entities(list(all_entities), n_clusters)

            logger.info(f"实体聚类结果，共 {len(entity_clusters)} 个簇:")
            for cluster_id, entities in entity_clusters.items():
                logger.info(f"簇 {cluster_id}: {entities}")

            # 4. 提取关系
            # relations_prompt = f"""
            # Task: Extract relationships between entities in the following text.
            #
            # Text:
            # {text_input}
            #
            # Entities list:
            # {list(all_entities)}
            #
            # IMPORTANT:
            # 1. ONLY create relationships between entities in the provided list above.
            # 2. DO NOT translate entity names. Keep them in English exactly as provided.
            # 3. Make relationship names short and descriptive (e.g., "studies", "works_at", "is_mother_of").
            #
            # Return the relationships as JSON array of triplets:
            # [["Entity1", "relationship", "Entity2"], ["Entity3", "relationship", "Entity4"], ...]
            # """

            relations_prompt = f"""
                        Task: Extract relationships between entities in the following text related to Military aviation and tactical aviation operations.

                        Text:
                        {text_input}

                        Entities list:
                        {list(all_entities)}

                        IMPORTANT:
                        1. Create relationships between entities in the provided list above as much as possible.
                        2. DO NOT translate entity names. Keep them as provided.
                        3. Focus on these types of relationships specific to Military aviation and tactical aviation operations:
                           - [物理与功能，例如：武器/设备_装配于_飞行器、系统_搭载于_平台、飞行器_发射_武器等]
                           - [战术与对抗，例如：传感器_探测_目标、电子战系统_干扰_雷达、飞行员_执行_任务等]
                           - [逻辑与依赖，例如：导弹_依赖_雷达、技术_支持_作战、环境_制约_作战、技术_影响_战术等] 
                           - [时间与空间，例如：事件_发生于_时间、任务_执行于_地点、飞行_持续_时间、装备_部署于_地区、武器/传感器_覆盖_区域等] 
                           - [人员与组织，例如：飞行员_隶属于_部队、技术人员_支持_作战、部队_执行_任务、飞行员_执行_任务等]
                           - [性能与参数，例如：飞行器_速度_性能、武器_射程_参数、传感器_精度_参数、飞行器_载荷_参数等]
                           - [对抗与竞争，例如：飞行器_对抗_敌方、部队_竞争_对手、技术_超越_对手、作战_制胜_对手等]
                           - [其他关系，例如：飞行器_维护_保养、技术_支持_作战、部队_执行_任务、飞行员_执行_任务等]
                           
                        4. Make relationship names short and descriptive.

                        Return the relationships as JSON array of triplets: 
                        [["Entity1", "relationship", "Entity2"], ["Entity3", "relationship", "Entity4"], ...]
                        """


            logger.info("提取关系中...")

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": relations_prompt,
                    "stream": False
                }
            )

            if response.status_code != 200:
                logger.error(f"关系API调用失败: {response.status_code} - {response.text}")
                return

            relations_content = response.json().get("response", "")

            # 提取JSON部分
            json_match = re.search(r'\[.*\]', relations_content, re.DOTALL)
            relations = set()

            if json_match:
                relations_json = json_match.group(0)
                relations_json = relations_json.replace("'", '"').replace('\n', '')
                try:
                    relations_list = json.loads(relations_json)
                    relations = {tuple(relation) for relation in relations_list if len(relation) == 3}
                except json.JSONDecodeError:
                    logger.warning("无法解析关系JSON，使用备用方法")
                    triple_matches = re.findall(r'\["([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\]', relations_json)
                    relations = {(s, p, o) for s, p, o in triple_matches}
            else:
                logger.warning("无法从响应中提取JSON，使用备用方法")
                triple_matches = re.findall(r'\["([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\]', relations_content)
                relations = {(s, p, o) for s, p, o in triple_matches}

            logger.info(f"提取到的关系: {relations}")

            # 5. 构建图谱
            edges = {relation[1] for relation in relations}

            # 构建知识图谱结构，包含聚类信息
            graph = {
                "entities": list(all_entities),
                "entity_clusters": {str(k): v for k, v in entity_clusters.items()},  # 转换键为字符串以便JSON序列化
                "edges": list(edges),
                "relations": [list(relation) for relation in relations]
            }

            # 打印生成的知识图谱
            print("\n生成的知识图谱:")
            print("实体 (Entities):", graph["entities"])
            print("实体聚类:")
            for cluster_id, entities in graph["entity_clusters"].items():
                print(f"  簇 {cluster_id}: {entities}")
            print("边 (Edges):", graph["edges"])
            print("关系 (Relations):", graph["relations"])

            # 保存知识图谱到JSON文件
            with open("knowledge_graph.json", "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.info("知识图谱已保存到 knowledge_graph.json")

        else:
            logger.error("没有提取到任何实体，无法构建图谱")
    except Exception as e:
        logger.error(f"错误发生: {e}")
        logger.error(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()

    # 将知识图谱存储到Neo4j
    if all_entities and relations:
        logger.info("开始将知识图谱存储到Neo4j...")
        store_in_neo4j(graph)
        logger.info("Neo4j存储过程完成")
    else:
        logger.warning("没有足够的数据来存储到Neo4j")


if __name__ == "__main__":
    main()