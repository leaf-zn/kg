import os
import logging
import json
import re
import requests
import numpy as np
import argparse
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


def read_text_from_file(file_path):
    """
    从文件中读取文本内容

    Args:
        file_path (str): 文本文件的路径

    Returns:
        str: 文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        logger.info(f"成功从文件 {file_path} 读取文本，长度：{len(text)} 字符")
        return text
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return ""


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
    entities_prompt = f"""
        Task: Extract ONLY the entities EXPLICITLY MENTIONED in the following text related to Military aviation and tactical aviation operations.
        
        Focus on extracting these specific types of entities:
        1. Aircraft types
        2. Weapon Systems
        3. Personnel roles
        4. Tactics and techniques
        5. Military units
        6. Sensors and electronic systems
        7. Performance Parameters
        
        IMPORTANT INSTRUCTIONS:
        - ONLY extract entities that are EXPLICITLY mentioned in the text.
        - DO NOT include examples that are not in the text.
        - DO NOT infer or generate entities based on domain knowledge.
        - DO NOT include entities from this instruction itself.
        
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



def extract_relations_from_chunk(base_url, model_name, chunk, chunk_entities):
    """从文本块中提取关系"""
    relations_prompt = f"""
        Task: Extract relationships between entities in the following text related to Military aviation and tactical aviation operations.

        Text:
        {chunk}

        Entities list:
        {list(chunk_entities)}

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
        return set()

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

    return relations

def cluster_relation_types(relation_types, n_clusters=3):
    """
    将关系类型聚类分组

    Args:
        relation_types (list): 关系类型列表
        n_clusters (int): 聚类数量

    Returns:
        dict: 聚类结果，键为簇ID，值为该簇中的关系类型列表
    """
    if len(relation_types) < n_clusters:
        logger.warning(f"关系类型数量({len(relation_types)})小于请求的聚类数量({n_clusters})，将调整聚类数量")
        n_clusters = max(1, len(relation_types) // 2)

    if len(relation_types) <= 1:
        return {0: relation_types}

    # 使用TF-IDF向量化关系类型
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform([str(relation) for relation in relation_types])

        # 如果关系类型太少，返回一个簇
        if X.shape[0] < n_clusters:
            return {0: relation_types}

        # 执行KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # 组织聚类结果
        clusters = {}
        for i, relation in enumerate(relation_types):
            cluster_id = kmeans.labels_[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(relation)

        return clusters
    except Exception as e:
        logger.error(f"关系聚类过程中出错: {e}")
        return {0: relation_types}


def main():
    # 命令行参数解析部分保持不变
    parser = argparse.ArgumentParser(description='军事航空知识图谱生成工具')
    parser.add_argument('--input', '-i', type=str, default='input_text.txt',
                        help='输入文本文件路径')
    parser.add_argument('--output', '-o', type=str, default='knowledge_graph.json',
                        help='输出知识图谱JSON文件路径')
    parser.add_argument('--model', '-m', type=str, default='qwen2.5:latest',
                        help='使用的LLM模型名称')
    parser.add_argument('--chunks', '-c', type=int, default=3,
                        help='每个块的句子数')
    parser.add_argument('--clusters', '-k', type=int, default=3,
                        help='实体聚类的数量')
    parser.add_argument('--relation-clusters', '-r', type=int, default=3,
                        help='关系聚类的数量')

    args = parser.parse_args()

    # Ollama API配置部分保持不变
    base_url = os.getenv("BASE_URL", "http://localhost:11434")
    logger.info(f"使用Ollama API基地址: {base_url}")

    # 测试API连接部分保持不变
    try:
        logger.info("测试API连接...")
        response = requests.get(f"{base_url}/api/version")
        logger.info(f"API连接测试结果: {response.status_code}")
        logger.info(f"API版本: {response.json()}")
    except Exception as e:
        logger.error(f"连接测试失败: {e}")
        return

    # 获取可用模型部分保持不变
    try:
        logger.info("获取可用模型...")
        response = requests.get(f"{base_url}/api/tags")
        models = response.json().get("models", [])
        logger.info(f"可用模型: {models}")

        # 确认模型是否可用
        model_name = args.model
        if not any(model.get("name") == model_name for model in models):
            logger.warning(f"未找到模型 {model_name}，将使用第一个可用模型")
            if models:
                model_name = models[0].get("name", "qwen2.5:latest")
    except Exception as e:
        logger.warning(f"获取模型列表失败: {e}，将使用默认模型: {args.model}")
        model_name = args.model

    logger.info(f"使用模型: {model_name}")

    # 从文件读取输入文本部分保持不变
    text_input = read_text_from_file(args.input)

    if not text_input:
        logger.error(f"无法读取输入文本，请检查文件路径: {args.input}")
        return

    # 生成知识图谱 - 开始修改
    try:
        logger.info("开始生成知识图谱...")

        # 1. 分块处理
        chunk_size = args.chunks
        chunks = chunk_text(text_input, chunk_size)
        
        # 创建存储所有块的实体和关系的集合
        all_entities = set()
        all_relations = set()
        
        # 2. 对每个块分别进行实体抽取和关系抽取
        for i, chunk in enumerate(chunks):
            logger.info(f"处理块 {i + 1}/{len(chunks)}...")
            
            # 2.1 从当前块中提取实体
            chunk_entities = extract_entities_from_chunk(base_url, model_name, chunk)
            logger.info(f"块 {i + 1} 中提取到的实体: {chunk_entities}")
            
            # 如果块中没有提取到实体，继续处理下一个块
            if not chunk_entities:
                logger.warning(f"块 {i + 1} 中没有提取到实体，跳过关系抽取")
                continue
                
            # 2.2 从当前块中提取关系
            chunk_relations = extract_relations_from_chunk(base_url, model_name, chunk, chunk_entities)
            logger.info(f"块 {i + 1} 中提取到的关系: {chunk_relations}")
            
            # 2.3 将当前块的实体和关系添加到总集合中
            all_entities.update(chunk_entities)
            all_relations.update(chunk_relations)
        
        logger.info(f"所有块中提取到的实体总数: {len(all_entities)}")
        logger.info(f"所有块中提取到的关系总数: {len(all_relations)}")
            
        if all_entities:
            # 3. 实体聚类
            n_clusters = min(args.clusters, len(all_entities))
            entity_clusters = cluster_entities(list(all_entities), n_clusters)
            
            logger.info(f"实体聚类结果，共 {len(entity_clusters)} 个簇:")
            for cluster_id, entities in entity_clusters.items():
                logger.info(f"簇 {cluster_id}: {entities}")
                
            # 4. 关系聚类 (新增)
            relation_types = {relation[1] for relation in all_relations}
            if relation_types:
                n_relation_clusters = min(args.relation_clusters, len(relation_types))
                relation_clusters = cluster_relation_types(list(relation_types), n_relation_clusters)
                
                logger.info(f"关系聚类结果，共 {len(relation_clusters)} 个簇:")
                for cluster_id, relations in relation_clusters.items():
                    logger.info(f"簇 {cluster_id}: {relations}")
            else:
                relation_clusters = {}
                
            # 5. 构建知识图谱
            edges = {relation[1] for relation in all_relations}
            
            # 构建知识图谱结构，包含聚类信息
            graph = {
                "entities": list(all_entities),
                "entity_clusters": {str(k): v for k, v in entity_clusters.items()},
                "relation_clusters": {str(k): v for k, v in relation_clusters.items()},  # 新增关系聚类
                "edges": list(edges),
                "relations": [list(relation) for relation in all_relations]
            }
            
            # 打印生成的知识图谱
            print("\n生成的知识图谱:")
            print("实体 (Entities):", graph["entities"])
            print("实体聚类:")
            for cluster_id, entities in graph["entity_clusters"].items():
                print(f"  簇 {cluster_id}: {entities}")
            print("关系聚类:")  # 新增
            for cluster_id, relations in graph["relation_clusters"].items():
                print(f"  簇 {cluster_id}: {relations}")
            print("边 (Edges):", graph["edges"])
            print("关系 (Relations):", graph["relations"])
            
            # 保存知识图谱到JSON文件
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.info(f"知识图谱已保存到 {args.output}")
            
            # 将知识图谱存储到Neo4j (保持不变)
            logger.info("开始将知识图谱存储到Neo4j...")
            store_in_neo4j(graph)
            logger.info("Neo4j存储过程完成")
        else:
            logger.error("没有提取到任何实体，无法构建图谱")
    except Exception as e:
        logger.error(f"错误发生: {e}")
        logger.error(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()