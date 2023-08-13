import spacy
from collections import defaultdict

# spaCy 모델 로드
nlp = spacy.load("en_core_web_sm")

# 문장 간의 유사도 계산 함수
def calculate_jaccard_similarity(doc1, doc2):
    tokens1 = set(token.text for token in doc1)
    tokens2 = set(token.text for token in doc2)

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)

# 유사한 문장끼리 묶어주는 함수
def group_similar_sentences(sentences, threshold=1.0):
    grouped_sentences = defaultdict(list)

    for i, sentence in enumerate(sentences):
        doc = nlp(sentence)
        grouped = False

        for group in grouped_sentences.values():
            for grouped_sentence in group:
                if calculate_jaccard_similarity(doc, nlp(grouped_sentence)) > threshold:
                    group.append(sentence)
                    grouped = True
                    break

        if not grouped:
            grouped_sentences[i].append(sentence)

            print(f"Processing sentence {i + 1}/{len(sentences)}", end="\r")

    return list(grouped_sentences.values())

def result():
    # 파일에서 문장들 읽어오기
    with open("questions.txt", "r", encoding="utf-8") as file:
        sentences = file.read().splitlines()
        print("문장 읽어오기 완료...")

    # 유사한 문장끼리 묶어주기
    similar_sentence_groups = group_similar_sentences(sentences, threshold=1.0)

    # 결과를 JaccardResult.txt 파일에 저장
    with open("JaccardResult.txt", "w", encoding="utf-8") as result_file:
        for i, group in enumerate(similar_sentence_groups, start=1):
            result_file.write(f"Group {i}:\n")
            for sentence in group:
                result_file.write(f"- {sentence}\n")
            result_file.write("\n")

    print("Results saved to JaccardResult.txt")

if __name__ == "__main__":
    result()
