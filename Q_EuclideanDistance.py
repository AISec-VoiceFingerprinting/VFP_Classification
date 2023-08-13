import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 특정한 토큰들에 대한 가중치 부여
def custom_tokenizer(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha]
    return tokens

# 유사한 문장끼리 묶어주는 함수
def group_similar_sentences(sentences, threshold=1.0):
    grouped_sentences = defaultdict(list)

    # TfidfVectorizer를 사용하여 문장을 벡터화
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
    X = vectorizer.fit_transform(sentences)

    # 유클리디언 유사도 계산
    euclidean_similarities = euclidean_distances(X)


    for i in range(len(sentences)):
        vec = X[i].toarray()
        added_to_group = False
        print(f"Processing sentence {i + 1}/{len(sentences)}", end="\r")

        for group_idx, group in grouped_sentences.items():
            similar_to_group = any(euclidean_similarities[i][sent_idx] <= threshold for sent_idx in group)
            if similar_to_group:
                group.append(i)
                added_to_group = True
                break

        if not added_to_group:
            grouped_sentences[i].append(i)

    return list(grouped_sentences.values())

def result():
    # 파일에서 문장들 읽어오기
    with open("questions.txt", "r", encoding="utf-8") as file:
        sentences = file.read().splitlines()
        print("문장 읽어오기 완료...")

    # 유사한 문장끼리 묶어주기
    similar_sentence_groups = group_similar_sentences(sentences, threshold=1.0)

    # 결과를 하나의 txt 파일에 저장
    with open("EuclideanResult.txt", "w", encoding="utf-8") as result_file:
        for i, group in enumerate(similar_sentence_groups, start=1):
            result_file.write(f"Group {i}:\n")
            for sent_idx in group:
                result_file.write(f"- {sentences[sent_idx]}\n")
            result_file.write("\n")

    print("Results saved to EuclideanResult.txt")

if __name__ == "__main__":
    result()
