import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pickle
pretrain_model_name = 'sentence-transformers/all-mpnet-base-v2'
semantic_embedding_dim = 768
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
similarity_threshold = 0.5


def generate_semantic_embeddings(dataset_type, news_category, mode):
    assert mode in ['corpus', 'full']
    store_root = dataset_type + '-SAG/semantic_embeddings' if mode == 'full' else dataset_type + '-SAG/corpus_semantic_embeddings'
    # 1. news meta information
    news_meta_file = '%s/news_meta-%s.json' % (store_root, news_category)
    if not os.path.exists(news_meta_file):
        news_title_dict = {}       # {news_ID: title}
        news_title_dict_inv = {}   # {title: [news_ID]}
        news_content_dict = {}     # {news_ID: content}
        news_content_dict_inv = {} # {content: [news_ID]}
        news_dict = {}             # {news_ID: index}
        news_dict_inv = {}         # {index: [news_ID]}
        titles = []                # [titles]
        contents = []              # [contents]
        empty_news_IDs = []
        news_ID_set = set()
        with open(dataset_type + '-SAG/news/%s.tsv' % news_category, 'r', encoding='utf-8') as f:
            for line in f:
                data_domain, news_ID, category, subCategory, title, content, _, title_entities, content_entities = line.split('\t')
                if dataset_type == 'small' and mode == 'corpus' and data_domain == 'test':
                    continue
                if news_ID not in news_ID_set:
                    news_ID_set.add(news_ID)
                    title = title.lower().replace('é', 'e')
                    content = content.lower().replace('é', 'e')
                    if title == '' and content != '':
                        title = content
                    elif title != '' and content == '':
                        content = title
                    elif title == '' and content == '':
                        empty_news_IDs.append(news_ID)
                        continue
                    news_title_dict[news_ID] = title
                    news_content_dict[news_ID] = content
                    if title not in news_title_dict_inv:
                        news_title_dict_inv[title] = [news_ID]
                    else:
                        news_title_dict_inv[title].append(news_ID)
                    if content not in news_content_dict_inv:
                        news_content_dict_inv[content] = [news_ID]
                    else:
                        news_content_dict_inv[content].append(news_ID)
        for i, title in enumerate(news_title_dict_inv):
            titles.append(title)
            news_dict_inv[str(i)] = []
            empty_content_flag = True
            for news_ID in news_title_dict_inv[title]:
                content = news_content_dict[news_ID]
                if content != '' and empty_content_flag:
                    contents.append(content)
                    empty_content_flag = False
                news_dict[news_ID] = i
                news_dict_inv[str(i)].append(news_ID)
            if empty_content_flag:
                contents.append(title)
        duplicated_content_set = set()
        content_set = set()
        for content in contents:
            if content in content_set:
                duplicated_content_set.add(content)
            content_set.add(content)
        for i in range(len(contents)):
            if contents[i] in duplicated_content_set:
                contents[i] = titles[i] + ' ' + contents[i]
        with open(news_meta_file, 'w', encoding='utf-8') as f:
            json.dump({
                'news_dict': news_dict,
                'news_dict_inv': news_dict_inv,
                'titles': titles,
                'contents': contents,
                'empty_news_IDs': empty_news_IDs
            }, f)
    else:
        with open(news_meta_file, 'r', encoding='utf-8') as f:
            MIND_news_info = json.load(f)
            news_dict = MIND_news_info['news_dict']
            news_dict_inv = MIND_news_info['news_dict_inv']
            titles = MIND_news_info['titles']
            contents = MIND_news_info['contents']
            empty_news_IDs = MIND_news_info['empty_news_IDs']
    # 2. semantic embeddings
    title_semantic_embedding_file = '%s/title_semantic_embeddings-%s.pkl' % (store_root, news_category)
    content_semantic_embedding_file = '%s/content_semantic_embeddings-%s.pkl' % (store_root, news_category)
    if not os.path.exists(title_semantic_embedding_file) or not os.path.exists(content_semantic_embedding_file):
        model = SentenceTransformer(pretrain_model_name)
        title_semantic_embeddings = torch.from_numpy(model.encode(titles))
        content_semantic_embeddings = torch.from_numpy(model.encode(contents))
        with open(title_semantic_embedding_file, 'wb') as f:
            pickle.dump(title_semantic_embeddings, f, protocol=4)
        with open(content_semantic_embedding_file, 'wb') as f:
            pickle.dump(content_semantic_embeddings, f, protocol=4)
    else:
        with open(title_semantic_embedding_file, 'rb') as f:
            title_semantic_embeddings = pickle.load(f)
        with open(content_semantic_embedding_file, 'rb') as f:
            content_semantic_embeddings = pickle.load(f)
    return news_dict, news_dict_inv, titles, contents, empty_news_IDs, title_semantic_embeddings, content_semantic_embeddings


def generate_cos_similarities(dataset_type, top_M, category, title_semantic_embeddings, content_semantic_embeddings, corpus_title_semantic_embeddings, corpus_content_semantic_embeddings):
    news_num = title_semantic_embeddings.size(0)
    corpus_news_num = corpus_title_semantic_embeddings.size(0)
    top_M = min(top_M, corpus_news_num - 1)
    title_cos_values_file = dataset_type + '-SAG/cos/title_cos_values-%d-%s.pkl' % (top_M, category)
    title_cos_indices_file = dataset_type + '-SAG/cos/title_cos_indices-%d-%s.pkl' % (top_M, category)
    content_cos_values_file = dataset_type + '-SAG/cos/content_cos_values-%d-%s.pkl' % (top_M, category)
    content_cos_indices_file = dataset_type + '-SAG/cos/content_cos_indices-%d-%s.pkl' % (top_M, category)
    title_content_cos_values_file = dataset_type + '-SAG/cos/title-content_cos_values-%d-%s.pkl' % (top_M, category)
    title_content_cos_indices_file = dataset_type + '-SAG/cos/title-content_cos_indices-%d-%s.pkl' % (top_M, category)
    content_title_cos_values_file = dataset_type + '-SAG/cos/content-title_cos_values-%d-%s.pkl' % (top_M, category)
    content_title_cos_indices_file = dataset_type + '-SAG/cos/content-title_cos_indices-%d-%s.pkl' % (top_M, category)
    average_cos_values_file = dataset_type + '-SAG/cos/average_cos_values-%d-%s.pkl' % (top_M, category)
    average_cos_indices_file = dataset_type + '-SAG/cos/average_cos_indices-%d-%s.pkl' % (top_M, category)
    if not os.path.exists(title_cos_values_file) or not os.path.exists(title_cos_indices_file) or not os.path.exists(content_cos_values_file) or not os.path.exists(content_cos_indices_file) or not os.path.exists(title_content_cos_values_file) or not os.path.exists(title_content_cos_indices_file) or not os.path.exists(content_title_cos_values_file) or not os.path.exists(content_title_cos_indices_file) or not os.path.exists(average_cos_values_file) or not os.path.exists(average_cos_indices_file):
        title_semantic_embeddings = title_semantic_embeddings.cuda()
        content_semantic_embeddings = content_semantic_embeddings.cuda()
        corpus_title_semantic_embeddings = corpus_title_semantic_embeddings.cuda()
        corpus_content_semantic_embeddings = corpus_content_semantic_embeddings.cuda()
        title_cos_values = torch.zeros([news_num, top_M+1], dtype=torch.float32, device=torch.device('cuda'))
        title_cos_indices = torch.zeros([news_num, top_M+1], dtype=torch.int32, device=torch.device('cuda'))
        content_cos_values = torch.zeros([news_num, top_M+1], dtype=torch.float32, device=torch.device('cuda'))
        content_cos_indices = torch.zeros([news_num, top_M+1], dtype=torch.int32, device=torch.device('cuda'))
        title_content_cos_values = torch.zeros([news_num, top_M+1], dtype=torch.float32, device=torch.device('cuda'))
        title_content_cos_indices = torch.zeros([news_num, top_M+1], dtype=torch.int32, device=torch.device('cuda'))
        content_title_cos_values = torch.zeros([news_num, top_M+1], dtype=torch.float32, device=torch.device('cuda'))
        content_title_cos_indices = torch.zeros([news_num, top_M+1], dtype=torch.int32, device=torch.device('cuda'))
        average_cos_values = torch.zeros([news_num, top_M+1], dtype=torch.float32, device=torch.device('cuda'))
        average_cos_indices = torch.zeros([news_num, top_M+1], dtype=torch.int32, device=torch.device('cuda'))
        cos = torch.nn.CosineSimilarity()
        with torch.no_grad():
            for i in range(news_num):
                title_title_cos = cos(title_semantic_embeddings[i].unsqueeze(dim=0).repeat(corpus_news_num, 1), corpus_title_semantic_embeddings)
                values, indices = torch.topk(title_title_cos, k=top_M+1)
                title_cos_values[i] = values
                title_cos_indices[i] = indices
                content_content_cos = cos(content_semantic_embeddings[i].unsqueeze(dim=0).repeat(corpus_news_num, 1), corpus_content_semantic_embeddings)
                values, indices = torch.topk(content_content_cos, k=top_M+1)
                content_cos_values[i] = values
                content_cos_indices[i] = indices
                title_content_cos = cos(title_semantic_embeddings[i].unsqueeze(dim=0).repeat(corpus_news_num, 1), corpus_content_semantic_embeddings)
                values, indices = torch.topk(title_content_cos, k=top_M+1)
                title_content_cos_values[i] = values
                title_content_cos_indices[i] = indices
                content_title_cos = cos(content_semantic_embeddings[i].unsqueeze(dim=0).repeat(corpus_news_num, 1), corpus_title_semantic_embeddings)
                values, indices = torch.topk(content_title_cos, k=top_M+1)
                content_title_cos_values[i] = values
                content_title_cos_indices[i] = indices
                values, indices = torch.topk((title_title_cos + content_content_cos + title_content_cos + content_title_cos) / 4, k=top_M+1)
                average_cos_values[i] = values
                average_cos_indices[i] = indices
        title_cos_values = title_cos_values.cpu()
        title_cos_indices = title_cos_indices.cpu()
        content_cos_values = content_cos_values.cpu()
        content_cos_indices = content_cos_indices.cpu()
        title_content_cos_values = title_content_cos_values.cpu()
        title_content_cos_indices = title_content_cos_indices.cpu()
        content_title_cos_values = content_title_cos_values.cpu()
        content_title_cos_indices = content_title_cos_indices.cpu()
        average_cos_values = average_cos_values.cpu()
        average_cos_indices = average_cos_indices.cpu()
        with open(title_cos_values_file, 'wb') as f:
            pickle.dump(title_cos_values, f)
        with open(title_cos_indices_file, 'wb') as f:
            pickle.dump(title_cos_indices, f)
        with open(content_cos_values_file, 'wb') as f:
            pickle.dump(content_cos_values, f)
        with open(content_cos_indices_file, 'wb') as f:
            pickle.dump(content_cos_indices, f)
        with open(title_content_cos_values_file, 'wb') as f:
            pickle.dump(title_content_cos_values, f)
        with open(title_content_cos_indices_file, 'wb') as f:
            pickle.dump(title_content_cos_indices, f)
        with open(content_title_cos_values_file, 'wb') as f:
            pickle.dump(content_title_cos_values, f)
        with open(content_title_cos_indices_file, 'wb') as f:
            pickle.dump(content_title_cos_indices, f)
        with open(average_cos_values_file, 'wb') as f:
            pickle.dump(average_cos_values, f)
        with open(average_cos_indices_file, 'wb') as f:
            pickle.dump(average_cos_indices, f)
    else:
        with open(title_cos_values_file, 'rb') as f:
            title_cos_values = pickle.load(f)
        with open(title_cos_indices_file, 'rb') as f:
            title_cos_indices = pickle.load(f)
        with open(content_cos_values_file, 'rb') as f:
            content_cos_values = pickle.load(f)
        with open(content_cos_indices_file, 'rb') as f:
            content_cos_indices = pickle.load(f)
        with open(title_content_cos_values_file, 'rb') as f:
            title_content_cos_values = pickle.load(f)
        with open(title_content_cos_indices_file, 'rb') as f:
            title_content_cos_indices = pickle.load(f)
        with open(content_title_cos_values_file, 'rb') as f:
            content_title_cos_values = pickle.load(f)
        with open(content_title_cos_indices_file, 'rb') as f:
            content_title_cos_indices = pickle.load(f)
        with open(average_cos_values_file, 'rb') as f:
            average_cos_values = pickle.load(f)
        with open(average_cos_indices_file, 'rb') as f:
            average_cos_indices = pickle.load(f)
    return title_cos_values, title_cos_indices, content_cos_values, content_cos_indices, title_content_cos_values, title_content_cos_indices, content_title_cos_values, content_title_cos_indices, average_cos_values, average_cos_indices


def generate_similariy_info(dataset_type, top_M, category, news_dict_inv, corpus_news_dict_inv, title_cos_values, title_cos_indices, content_cos_values, content_cos_indices, title_content_cos_values, title_content_cos_indices, content_title_cos_values, content_title_cos_indices, average_cos_values, average_cos_indices):
    assert len(news_dict_inv) == title_cos_values.size(0)
    news_num = len(news_dict_inv)
    corpus_news_num = len(corpus_news_dict_inv)
    top_M = min(top_M, corpus_news_num - 1)
    title_similarity_file = dataset_type + '-SAG/similarity/title_similarity-%d-%s.json' % (top_M, category)
    content_similarity_file = dataset_type + '-SAG/similarity/content_similarity-%d-%s.json' % (top_M, category)
    title_content_similarity_file = dataset_type + '-SAG/similarity/title-content_similarity-%d-%s.json' % (top_M, category)
    content_title_similarity_file = dataset_type + '-SAG/similarity/content-title_similarity-%d-%s.json' % (top_M, category)
    average_similarity_file = dataset_type + '-SAG/similarity/average_similarity-%d-%s.json' % (top_M, category)
    if not os.path.exists(title_similarity_file) or not os.path.exists(content_similarity_file) or not os.path.exists(title_content_similarity_file) or not os.path.exists(content_title_similarity_file) or not os.path.exists(average_similarity_file):
        title_similarity = {}
        content_similarity = {}
        title_content_similarity = {}
        content_title_similarity = {}
        average_similarity = {}
        title_cos_values = title_cos_values.tolist()
        title_cos_indices = title_cos_indices.tolist()
        content_cos_values = content_cos_values.tolist()
        content_cos_indices = content_cos_indices.tolist()
        title_content_cos_values = title_content_cos_values.tolist()
        title_content_cos_indices = title_content_cos_indices.tolist()
        content_title_cos_values = content_title_cos_values.tolist()
        content_title_cos_indices = content_title_cos_indices.tolist()
        average_cos_values = average_cos_values.tolist()
        average_cos_indices = average_cos_indices.tolist()
        for i in range(news_num):
            # title-tille similarity
            values = title_cos_values[i]
            indices = title_cos_indices[i]
            title_similarity[i] = [{'ID': indices[j], 'cos_similarity': values[j]} for j in range(top_M+1)]
            # content-content similarity
            values = content_cos_values[i]
            indices = content_cos_indices[i]
            content_similarity[i] = [{'ID': indices[j], 'cos_similarity': values[j]} for j in range(top_M+1)]
            # title-content similarity
            values = title_content_cos_values[i]
            indices = title_content_cos_indices[i]
            title_content_similarity[i] = [{'ID': indices[j], 'cos_similarity': values[j]} for j in range(top_M+1)]
            # content-title similarity
            values = content_title_cos_values[i]
            indices = content_title_cos_indices[i]
            content_title_similarity[i] = [{'ID': indices[j], 'cos_similarity': values[j]} for j in range(top_M+1)]
            # average similarity
            values = average_cos_values[i]
            indices = average_cos_indices[i]
            average_similarity[i] = [{'ID': indices[j], 'cos_similarity': values[j]} for j in range(top_M+1)]
        with open(title_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(title_similarity, f)
        with open(content_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(content_similarity, f)
        with open(title_content_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(title_content_similarity, f)
        with open(content_title_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(content_title_similarity, f)
        with open(average_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(average_similarity, f)
    else:
        with open(title_similarity_file, 'r', encoding='utf-8') as f:
            title_similarity = json.load(f)
        with open(content_similarity_file, 'r', encoding='utf-8') as f:
            content_similarity = json.load(f)
        with open(title_content_similarity_file, 'r', encoding='utf-8') as f:
            title_content_similarity = json.load(f)
        with open(content_title_similarity_file, 'r', encoding='utf-8') as f:
            content_title_similarity = json.load(f)
        with open(average_similarity_file, 'r', encoding='utf-8') as f:
            average_similarity = json.load(f)
    return title_similarity, content_similarity, title_content_similarity, content_title_similarity, average_similarity


def generate_similar_news_list(dataset_type, top_M, category, news_dict_inv, corpus_news_dict_inv, empty_news_IDs, title_similarity, content_similarity, title_content_similarity, content_title_similarity, average_similarity):
    assert len(news_dict_inv) == len(title_similarity)
    news_num = len(news_dict_inv)
    corpus_news_num = len(corpus_news_dict_inv)
    top_M = min(top_M, corpus_news_num - 1)
    news_title_similarity_file = dataset_type + '-SAG/news_similarity/news_title_similarity-%d-%s.json' % (top_M, category)
    news_content_similarity_file = dataset_type + '-SAG/news_similarity/news_content_similarity-%d-%s.json' % (top_M, category)
    news_title_content_similarity_file = dataset_type + '-SAG/news_similarity/news_title-content_similarity-%d-%s.json' % (top_M, category)
    news_content_title_similarity_file = dataset_type + '-SAG/news_similarity/news_content-title_similarity-%d-%s.json' % (top_M, category)
    news_average_similarity_file = dataset_type + '-SAG/news_similarity/news_average_similarity-%d-%s.json' % (top_M, category)
    if not os.path.exists(news_title_similarity_file) or not os.path.exists(news_content_similarity_file) or not os.path.exists(news_title_content_similarity_file) or not os.path.exists(news_content_title_similarity_file) or not os.path.exists(news_average_similarity_file):
        news_title_similarity = {}
        news_content_similarity = {}
        news_title_content_similarity = {}
        news_content_title_similarity = {}
        news_average_similarity = {}
        for index in news_dict_inv:
            for news_ID in news_dict_inv[index]:
                # 1. title similarity
                cnt = 0
                similarity_news = []
                similarity = title_similarity[int(index)]
                for i in range(top_M+1):
                    corpus_news_IDs = corpus_news_dict_inv[str(similarity[i]['ID'])]
                    if news_ID not in corpus_news_IDs:
                        similarity_news.append({
                            'news_ID': corpus_news_IDs[0],
                            'cos_similarity': similarity[i]['cos_similarity']
                        })
                        cnt += 1
                        if cnt == top_M:
                            break
                news_title_similarity[news_ID] = similarity_news
                # 2. content similarity
                cnt = 0
                similarity_news = []
                similarity = content_similarity[int(index)]
                for i in range(top_M+1):
                    corpus_news_IDs = corpus_news_dict_inv[str(similarity[i]['ID'])]
                    if news_ID not in corpus_news_IDs:
                        similarity_news.append({
                            'news_ID': corpus_news_IDs[0],
                            'cos_similarity': similarity[i]['cos_similarity']
                        })
                        cnt += 1
                        if cnt == top_M:
                            break
                news_content_similarity[news_ID] = similarity_news
                # 3. title-content similarity
                cnt = 0
                similarity_news = []
                similarity = title_content_similarity[int(index)]
                for i in range(top_M+1):
                    corpus_news_IDs = corpus_news_dict_inv[str(similarity[i]['ID'])]
                    if news_ID not in corpus_news_IDs:
                        similarity_news.append({
                            'news_ID': corpus_news_IDs[0],
                            'cos_similarity': similarity[i]['cos_similarity']
                        })
                        cnt += 1
                        if cnt == top_M:
                            break
                news_title_content_similarity[news_ID] = similarity_news
                # 4. content-title similarity
                cnt = 0
                similarity_news = []
                similarity = content_title_similarity[int(index)]
                for i in range(top_M+1):
                    corpus_news_IDs = corpus_news_dict_inv[str(similarity[i]['ID'])]
                    if news_ID not in corpus_news_IDs:
                        similarity_news.append({
                            'news_ID': corpus_news_IDs[0],
                            'cos_similarity': similarity[i]['cos_similarity']
                        })
                        cnt += 1
                        if cnt == top_M:
                            break
                news_content_title_similarity[news_ID] = similarity_news
                # 5. average similarity
                cnt = 0
                similarity_news = []
                similarity = average_similarity[int(index)]
                for i in range(top_M+1):
                    corpus_news_IDs = corpus_news_dict_inv[str(similarity[i]['ID'])]
                    if news_ID not in corpus_news_IDs:
                        similarity_news.append({
                            'news_ID': corpus_news_IDs[0],
                            'cos_similarity': similarity[i]['cos_similarity']
                        })
                        cnt += 1
                        if cnt == top_M:
                            break
                news_average_similarity[news_ID] = similarity_news
        candidates = []
        with open(dataset_type + '-SAG/news/%s.tsv' % category, 'r', encoding='utf-8') as f:
            for line in f:
                data_domain, news_ID, category, subCategory, title, content, _, title_entities, content_entities = line.split('\t')
                candidates.append(news_ID)
        for news_ID in empty_news_IDs:
            indices = np.random.choice(len(candidates), top_M+1, replace=False)
            cnt = 0
            similarity_candidates = []
            for index in indices:
                if candidates[index] != news_ID:
                    similarity_candidates.append({'news_ID': candidates[index], 'cos_similarity': 0})
                    cnt += 1
                    if cnt == top_M:
                        break
            news_title_similarity[news_ID] = similarity_candidates
            news_content_similarity[news_ID] = similarity_candidates
            news_title_content_similarity[news_ID] = similarity_candidates
            news_content_title_similarity[news_ID] = similarity_candidates
            news_average_similarity[news_ID] = similarity_candidates
        with open(news_title_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(news_title_similarity, f)
        with open(news_content_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(news_content_similarity, f)
        with open(news_title_content_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(news_title_content_similarity, f)
        with open(news_content_title_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(news_content_title_similarity, f)
        with open(news_average_similarity_file, 'w', encoding='utf-8') as f:
            json.dump(news_average_similarity, f)
    else:
        with open(news_title_similarity_file, 'r', encoding='utf-8') as f:
            news_title_similarity = json.load(f)
        with open(news_content_similarity_file, 'r', encoding='utf-8') as f:
            news_content_similarity = json.load(f)
        with open(news_title_content_similarity_file, 'r', encoding='utf-8') as f:
            news_title_content_similarity = json.load(f)
        with open(news_content_title_similarity_file, 'r', encoding='utf-8') as f:
            news_content_title_similarity = json.load(f)
        with open(news_average_similarity_file, 'r', encoding='utf-8') as f:
            news_average_similarity = json.load(f)
    return news_title_similarity, news_content_similarity, news_title_content_similarity, news_content_title_similarity, news_average_similarity


def aggregate(dataset_type, top_M, news_category_dict):
    similarity_file = dataset_type + '-SAG/similarity-%d.json' % top_M
    if not os.path.exists(similarity_file):
        news_similarity_dict = {}
        for sub_similarity_file in os.listdir(dataset_type + '-SAG/news_similarity'):
            if 'news_average_similarity-' in sub_similarity_file:
                with open(dataset_type + '-SAG/news_similarity/' + sub_similarity_file, 'r') as f:
                    similarity_dict = json.load(f)
                    for news_ID in similarity_dict:
                        similarity = similarity_dict[news_ID]
                        news_similarity_dict[news_ID] = [[s['news_ID'], s['cos_similarity']] for s in similarity]
        with open('news_ID-' + dataset_type + '.json', 'r', encoding='utf-8') as f:
            news_ID_dict = json.load(f)
            for news_ID in news_ID_dict:
                if news_ID not in news_similarity_dict:
                    news_similarity_dict[news_ID] = []
        with open(similarity_file, 'w', encoding='utf-8') as f:
            json.dump(news_similarity_dict, f)
    else:
        with open(similarity_file, 'r', encoding='utf-8') as f:
            news_similarity_dict = json.load(f)
    return news_similarity_dict


def generate_news_graph(dataset_type, news_similarity_dict, news_ID_dict, top_M, hop, news_node_num):
    news_num = len(news_ID_dict)
    news_ID_dict_inv = {news_ID_dict[news_ID]: news_ID for news_ID in news_ID_dict}
    news_node_ID = np.zeros([news_num, news_node_num], dtype=np.int32)
    news_graph = np.zeros([news_num, news_node_num, news_node_num], dtype=bool)
    news_graph_mask = np.zeros([news_num, news_node_num], dtype=bool)
    news_graph_mask[:, 0] = 1
    for i in range(1, news_num):
        # BFS
        news_node_ID[i, 0] = i
        node_pos = {i: 0}
        depths = [0 for _ in range(news_node_num)]
        head = 0
        rear = 1
        while head < rear:
            if depths[head] == hop:
                head += 1
                continue
            news_index = news_node_ID[i, head]
            for index, (_news_ID, cos_similarity) in enumerate(news_similarity_dict[news_ID_dict_inv[news_index]]):
                if depths[head] > 0 and (cos_similarity < similarity_threshold or index == top_M - 1):
                    break
                _news_index = news_ID_dict[_news_ID]
                if _news_index not in node_pos:
                    news_node_ID[i, rear] = _news_index
                    news_graph_mask[i, rear] = 1
                    node_pos[_news_index] = rear
                    news_graph[i, head, rear] = 1
                    news_graph[i, rear, head] = 1
                    depths[rear] = depths[head] + 1
                    rear += 1
                else:
                    pos = node_pos[_news_index]
                    news_graph[i, head, pos] = 1
                    news_graph[i, pos, head] = 1
            head += 1
    return news_node_ID, news_graph, news_graph_mask


def visualize(dataset_type, news_ID_dict, train_root, dev_root, test_root, news_node_ID, news_graph, news_node_num, top_M, hop, suffix=''):
    news_ID_dict_inv = {news_ID_dict[news_ID]: news_ID for news_ID in news_ID_dict}
    title_dict = {}
    for root in [train_root, dev_root, test_root]:
        with open(os.path.join(root, 'news.tsv'), 'r', encoding='utf-8') as news_f:
            for line in news_f:
                news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
                if news_ID not in title_dict:
                    title_dict[news_ID] = title.lower().replace('é', 'e')
    with open(dataset_type + '-SAG/visualize-%d-%d%s.tsv' % (top_M, hop, suffix), 'w', encoding='utf-8') as f:
        f.write('Node1\tNode2\tTitle1\tTitle2\n')
        for i in range(news_node_num):
            if news_node_ID[i] == 0:
                break
            for j in range(news_node_num):
                if news_graph[i, j] == 1:
                    f.write('%d\t%d\t%s\t%s\n' % (i, j, title_dict[news_ID_dict_inv[news_node_ID[i]]], title_dict[news_ID_dict_inv[news_node_ID[j]]]))
        f.write('\nnews graph\n')
        for i in range(news_node_num):
            for j in range(news_node_num):
                f.write(str(int(news_graph[i, j])))
                if j != news_node_num - 1:
                    f.write('\t')
                else:
                    f.write('\n')


def construct_SAG(dataset_type, train_root, dev_root, test_root, top_M, hop, news_node_num, news_ID_dict):
    if not os.path.exists(dataset_type + '-SAG'):
        os.mkdir(dataset_type + '-SAG')
    if not os.path.exists(dataset_type + '-SAG/news'):
        os.mkdir(dataset_type + '-SAG/news')
    if not os.path.exists(dataset_type + '-SAG/semantic_embeddings'):
        os.mkdir(dataset_type + '-SAG/semantic_embeddings')
    if not os.path.exists(dataset_type + '-SAG/corpus_semantic_embeddings'):
        os.mkdir(dataset_type + '-SAG/corpus_semantic_embeddings')
    if not os.path.exists(dataset_type + '-SAG/cos'):
        os.mkdir(dataset_type + '-SAG/cos')
    if not os.path.exists(dataset_type + '-SAG/similarity'):
        os.mkdir(dataset_type + '-SAG/similarity')
    if not os.path.exists(dataset_type + '-SAG/news_similarity'):
        os.mkdir(dataset_type + '-SAG/news_similarity')
    with open('category-' + dataset_type + '.json', 'r', encoding='utf-8') as f:
        news_category_dict = json.load(f)
    news_files = [dataset_type + '-SAG/news/%s.tsv' % category for category in news_category_dict]
    non_empty_corpus = {category : False for category in news_category_dict}
    if not all(list(map(os.path.exists, news_files))):
        news_fs = [open(news_file, 'w', encoding='utf-8') for news_file in news_files]
        news_ID_set = set()
        for i, root in enumerate([train_root, dev_root, test_root]):
            with open(os.path.join(root, 'news.tsv'), 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        news_ID, category, subCategory, title, content, _, title_entities, content_entities = line.split('\t')
                        if news_ID not in news_ID_set:
                            news_ID_set.add(news_ID)
                            if i < 2:
                                news_fs[news_category_dict[category]].write('train_dev\t' + line)
                                non_empty_corpus[category] = True
                            else:
                                news_fs[news_category_dict[category]].write('test\t' + line)
        for news_f in news_fs:
            news_f.close()
    else:
        for i, root in enumerate([train_root, dev_root, test_root]):
            with open(os.path.join(root, 'news.tsv'), 'r', encoding='utf-8') as f:
                for line in f:
                    if len(line.strip()) > 0:
                        news_ID, category, subCategory, title, content, _, title_entities, content_entities = line.split('\t')
                        if i < 2:
                            non_empty_corpus[category] = True

    for category in news_category_dict:
        if non_empty_corpus[category]:
            news_dict1, news_dict_inv1, titles1, contents1, empty_news_IDs1, title_semantic_embeddings1, content_semantic_embeddings1 = generate_semantic_embeddings(dataset_type, category, mode='full')
            news_dict2, news_dict_inv2, titles2, contents2, empty_news_IDs2, title_semantic_embeddings2, content_semantic_embeddings2 = generate_semantic_embeddings(dataset_type, category, mode='corpus')
            title_cos_values, title_cos_indices, content_cos_values, content_cos_indices, title_content_cos_values, title_content_cos_indices, content_title_cos_values, content_title_cos_indices, average_cos_values, average_cos_indices = generate_cos_similarities(dataset_type, top_M, category, title_semantic_embeddings1, content_semantic_embeddings1, title_semantic_embeddings2, content_semantic_embeddings2)
            title_similarity, content_similarity, title_content_similarity, content_title_similarity, average_similarity = generate_similariy_info(dataset_type, top_M, category, news_dict_inv1, news_dict_inv2, title_cos_values, title_cos_indices, content_cos_values, content_cos_indices, title_content_cos_values, title_content_cos_indices, content_title_cos_values, content_title_cos_indices, average_cos_values, average_cos_indices)
            generate_similar_news_list(dataset_type, top_M, category, news_dict_inv1, news_dict_inv2, empty_news_IDs1, title_similarity, content_similarity, title_content_similarity, content_title_similarity, average_similarity)
    news_similarity_dict = aggregate(dataset_type, top_M, news_category_dict)
    assert len(news_similarity_dict) == len(news_ID_dict), str(len(news_similarity_dict)) + ' : ' + str(len(news_ID_dict))
    news_node_ID, news_graph, news_graph_mask = generate_news_graph(dataset_type, news_similarity_dict, news_ID_dict, top_M, hop, news_node_num)
    return news_node_ID, news_graph, news_graph_mask
