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
        neo4j_password = "982138783wjk."

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
    entities_prompt = f"""
    Task: Extract all important entities (people, places, organizations, concepts) from the following text.
    
    DO NOT translate entity names. Keep them in their original language (English).
    
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
        model_name = "qwen2.5"
        if not any(model.get("name") == model_name for model in models):
            logger.warning(f"未找到模型 {model_name}，将使用第一个可用模型")
            if models:
                model_name = models[0].get("name", "qwen2.5")
    except Exception as e:
        logger.warning(f"获取模型列表失败: {e}，将使用默认模型: qwen2.5")
        model_name = "qwen2.5"

    logger.info(f"使用模型: {model_name}")

    # 输入文本 - 可以是更长的文本
    text_input = """
    Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father.
    Josh is a student at XYZ University. He studies computer science.
    Andrew works as a software engineer at ABC Corporation.
    Linda is a teacher at a local high school.
    The XYZ University has a renowned Computer Science department.
    Computer Science is a field that studies computation and information processing.
    ABC Corporation is a leading technology company in the software industry.
    The software industry creates and maintains software applications and systems.
    High schools provide education for students typically between the ages of 14 and 18.
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
            relations_prompt = f"""
            Task: Extract relationships between entities in the following text.

            Text:
            {text_input}

            Entities list:
            {list(all_entities)}

            IMPORTANT:
            1. ONLY create relationships between entities in the provided list above.
            2. DO NOT translate entity names. Keep them in English exactly as provided.
            3. Make relationship names short and descriptive (e.g., "studies", "works_at", "is_mother_of").

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