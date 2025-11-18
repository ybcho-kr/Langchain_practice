"""
유틸리티 헬퍼 함수
공통으로 사용되는 유틸리티 함수들
"""

import re
from typing import List, Tuple


def _extract_korean_chars(text: str) -> str:
    """한글 문자만 추출"""
    return ''.join(re.findall(r'[가-힣]', text))


def _has_question_particle(text: str) -> bool:
    """질문 어미 확인 (한국어 질문 표시)"""
    question_particles = ['까', '지', '나', '니', '인가', '인지', '뭔가', '뭔지']
    # 마지막 문자나 어미 확인
    if any(text.endswith(part) for part in question_particles):
        return True
    # 질문어 포함
    question_words = ['무엇', '어떻게', '왜', '언제', '어디서', '누가', '어떤', '몇', '얼마']
    return any(word in text for word in question_words)


def _calculate_general_score(question: str, question_lower: str) -> Tuple[float, List[str]]:
    """
    일반 질문일 가능성을 점수로 계산
    
    Returns:
        (점수, 판별 근거 리스트)
    """
    score = 0.0
    reasons = []
    
    # === 일반 대화 키워드 ===
    greeting_patterns = [
        (['안녕', '안녕하세요', '안녕히', '하이', '헬로'], 10.0, '인사말'),
        (['고마워', '고맙습니다', '감사', '감사합니다', 'thank', 'thanks'], 8.0, '감사 표현'),
        (['반가워', '반갑습니다', '만나서'], 7.0, '만남 표현'),
        (['잘있어', '잘다녀', '잘가', '잘다녀와', '수고', '수고하셨어요'], 7.0, '작별 인사'),
        (['뭐해', '뭐하니', '뭐하고있어', '어떻게지내', '잘지내', '어떠니'], 6.0, '일상 대화'),
        (['좋은하루', '좋은밤', '좋은아침', '좋은점심', '좋은저녁'], 6.0, '시간 인사'),
        (['hello', 'hi', 'hey', 'bye', 'goodbye', 'see you'], 8.0, '영어 인사'),
    ]
    
    for keywords, weight, reason in greeting_patterns:
        if any(keyword in question_lower for keyword in keywords):
            score += weight
            reasons.append(reason)
    
    # === 정규표현식 패턴 ===
    # 간단한 인사 패턴
    greeting_patterns_regex = [
        (r'^안녕', 9.0, '안녕 시작'),
        (r'고마워?$', 8.0, '고마워로 끝'),
        (r'^[hH]i', 7.0, 'Hi 시작'),
        (r'^[hH]ello', 8.0, 'Hello 시작'),
        (r'^\?$', 3.0, '단순 물음표'),
    ]
    
    for pattern, weight, reason in greeting_patterns_regex:
        if re.search(pattern, question):
            score += weight
            reasons.append(reason)
    
    # === 문장 구조 분석 ===
    # 매우 짧은 문장 (3자 이하)
    if len(_extract_korean_chars(question)) <= 3 and not _has_question_particle(question):
        score += 5.0
        reasons.append('매우 짧은 문장')
    
    # 질문어 없이 매우 짧음 (5자 이하)
    if len(question) <= 5 and not _has_question_particle(question):
        score += 4.0
        reasons.append('짧은 문장')
    
    # === 일반 대화 표현 ===
    casual_expressions = [
        '오늘', '어제', '내일', '지금', '요즘', '요새', '요전에',
        '어떻게', '왜요', '뭐요', '뭔가요', '그렇군요', '맞아요',
        '그래요', '그래', '좋아요', '좋아', '싫어요', '싫어'
    ]
    
    if any(expr in question_lower for expr in casual_expressions):
        # 전문 키워드와 함께 나오지 않았을 때만
        if not any(prof in question_lower for prof in ['진단', '분석', '측정', '변압기', '전기']):
            score += 3.0
            reasons.append('일상 표현')
    
    # === 점수 조정 (전문 키워드 감점) ===
    professional_keywords_strong = [
        '변압기', '전기', '설비', '진단', 'dga', 'pd', '절연', '전압', '전류',
        '분석', '측정', '기준', '판정', '시험', '점검', '고장', '결함',
        '코어', '권선', '냉각', '절연유', '가스', '방전', '절연체',
        '케이블', '배전', '송전', '개폐기', '차단기', '비전압', '전압강하',
        '전력', '피상전력', '유효전력', '무효전력', '역률', '임피던스'
    ]
    
    professional_keywords_weak = [
        '방법', '과정', '절차', '원리', '이유', '원인', '효과', '영향',
        '개선', '대책', '조치', '방안', '솔루션'
    ]
    
    # 강한 전문 키워드가 있으면 즉시 전문 질문으로 판정
    for keyword in professional_keywords_strong:
        if keyword in question_lower:
            score -= 100.0  # 강력한 감점
            reasons.append(f'전문 용어: {keyword}')
    
    # 약한 전문 키워드는 문맥에 따라
    weak_professional_count = sum(1 for keyword in professional_keywords_weak if keyword in question_lower)
    if weak_professional_count > 0:
        # 질문 구조가 있을 때만 전문 질문 가능성
        if _has_question_particle(question) or len(question) > 10:
            score -= weak_professional_count * 5.0
            reasons.append(f'전문 질문어 {weak_professional_count}개')
    
    # === 영어 전문 용어 체크 ===
    english_technical = [
        'voltage', 'current', 'transformer', 'insulation', 'diagnosis',
        'analysis', 'measurement', 'test', 'fault', 'failure'
    ]
    
    for term in english_technical:
        if term in question_lower:
            score -= 15.0
            reasons.append(f'영어 전문 용어: {term}')
    
    # === 숫자나 단위 포함 시 전문 질문 가능성 증가 ===
    if re.search(r'\d+[kmKM㎊㎌%℃℉VVAkW㎾]', question):
        score -= 5.0
        reasons.append('숫자 및 단위 포함')
    
    return score, reasons


def is_general_question(question: str, threshold: float = 5.0) -> bool:
    """
    일반적인 질문(인사말 등)인지 판별 (개선된 버전)
    
    Args:
        question: 판별할 질문 문자열
        threshold: 일반 질문 판별 임계값 (기본 5.0)
        
    Returns:
        일반 질문이면 True, 전문 질문이면 False
    """
    question = question.strip()
    if not question:
        return False
    
    question_lower = question.lower()
    
    # 점수 기반 판별
    score, reasons = _calculate_general_score(question, question_lower)
    
    # 점수가 임계값보다 높으면 일반 질문
    is_general = score >= threshold
    
    return is_general


def get_question_analysis(question: str) -> dict:
    """
    질문 분석 결과 반환 (디버깅 및 확장용)
    
    Returns:
        판별 결과, 점수, 판별 근거 등
    """
    question = question.strip()
    if not question:
        return {
            'is_general': False,
            'score': 0.0,
            'reasons': [],
            'confidence': 0.0
        }
    
    question_lower = question.lower()
    score, reasons = _calculate_general_score(question, question_lower)
    is_general = score >= 5.0
    
    # 신뢰도 계산 (절댓값으로 높을수록 확실)
    confidence = min(abs(score) / 20.0, 1.0)
    
    return {
        'is_general': is_general,
        'score': score,
        'reasons': reasons,
        'confidence': confidence,
        'question_length': len(question),
        'korean_char_count': len(_extract_korean_chars(question)),
        'has_question_particle': _has_question_particle(question)
    }

