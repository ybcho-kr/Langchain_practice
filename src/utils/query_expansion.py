"""
쿼리 확장 모듈
kiwipiepy 형태소 분석기 기반 쿼리 확장 및 검색 범위 확대
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import get_logger

try:
    from kiwipiepy import Kiwi
    KIWIPIEPY_AVAILABLE = True
except ImportError:
    KIWIPIEPY_AVAILABLE = False
    Kiwi = None


@dataclass
class SynonymEntry:
    """동의어 항목"""
    term: str
    synonyms: List[str]
    category: str  # 'technical', 'general', 'domain'
    pos: Optional[str] = None  # 품사 정보 (NNG, NNP 등)


class QueryExpander:
    """쿼리 확장기 (kiwipiepy 기반)"""
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None, use_morphological: bool = True):
        """
        Args:
            synonym_dict: 동의어 사전 (None이면 기본 사전 사용)
            use_morphological: 형태소 분석 사용 여부 (기본값: True)
        """
        self.logger = get_logger()
        self.use_morphological = use_morphological and KIWIPIEPY_AVAILABLE
        
        # Kiwi 인스턴스 초기화
        if self.use_morphological:
            try:
                self.kiwi = Kiwi()
                self._add_domain_words()  # 도메인 용어 추가
                self.logger.info("kiwipiepy 형태소 분석기 초기화 완료")
            except Exception as e:
                self.logger.warning(f"kiwipiepy 초기화 실패, 정규식 방식으로 폴백: {str(e)}")
                self.use_morphological = False
                self.kiwi = None
        else:
            self.kiwi = None
            if not KIWIPIEPY_AVAILABLE:
                self.logger.warning("kiwipiepy가 설치되지 않았습니다. 정규식 방식으로 동작합니다.")
        
        # 동의어 사전 초기화 (기존 구조와 호환성 유지)
        self.synonym_dict = synonym_dict or self._load_default_synonyms()
        
        # 품사별 동의어 사전 (형태소 분석 사용 시)
        self.pos_synonym_dict: Dict[str, Dict[str, List[str]]] = {}
        if self.use_morphological:
            self._build_pos_synonym_dict()
    
    def _add_domain_words(self):
        """전기설비 진단 도메인 용어 추가"""
        if not self.kiwi:
            return
        
        domain_words = [
            # 기술 용어 (일반명사)
            ("변압기", "NNG", 0.0),
            ("절연저항", "NNG", 0.0),
            ("절연유", "NNG", 0.0),
            ("절연오일", "NNG", 0.0),
            ("트랜스오일", "NNG", 0.0),
            ("절연액", "NNG", 0.0),
            ("절연능력", "NNG", 0.0),
            ("절연상태", "NNG", 0.0),
            ("변압장치", "NNG", 0.0),
            ("트랜스포머", "NNG", 0.0),
            ("트랜스", "NNG", 0.0),
            
            # 진단 용어
            ("진단", "NNG", 0.0),
            ("점검", "NNG", 0.0),
            ("검사", "NNG", 0.0),
            ("평가", "NNG", 0.0),
            ("검증", "NNG", 0.0),
            ("체크", "NNG", 0.0),
            
            # 측정 용어
            ("측정", "NNG", 0.0),
            ("검측", "NNG", 0.0),
            ("계측", "NNG", 0.0),
            
            # 기준/규격 용어
            ("기준", "NNG", 0.0),
            ("규격", "NNG", 0.0),
            ("표준", "NNG", 0.0),
            ("기준값", "NNG", 0.0),
            ("기준치", "NNG", 0.0),
            
            # 고유명사 (약어 등)
            ("DGA", "NNP", 0.0),
        ]
        
        try:
            for word, pos, score in domain_words:
                self.kiwi.add_user_word(word, pos, score)
            self.logger.debug(f"도메인 용어 {len(domain_words)}개 추가 완료")
        except Exception as e:
            self.logger.warning(f"도메인 용어 추가 중 일부 실패: {str(e)}")
    
    def _build_pos_synonym_dict(self):
        """품사별 동의어 사전 구축"""
        # 기존 동의어 사전을 품사별로 변환
        for term, synonyms in self.synonym_dict.items():
            # 형태소 분석으로 품사 확인
            try:
                morphs = self.kiwi.analyze(term)
                if morphs:
                    # 첫 번째 분석 결과의 품사 사용
                    pos = morphs[0][1]  # (형태소, 품사, 시작, 끝)
                    if pos.startswith('NN'):  # 명사만 처리
                        if pos not in self.pos_synonym_dict:
                            self.pos_synonym_dict[pos] = {}
                        self.pos_synonym_dict[pos][term] = synonyms
            except Exception:
                # 분석 실패 시 기본 품사(NNG)로 처리
                if 'NNG' not in self.pos_synonym_dict:
                    self.pos_synonym_dict['NNG'] = {}
                self.pos_synonym_dict['NNG'][term] = synonyms
    
    def _load_default_synonyms(self) -> Dict[str, List[str]]:
        """기본 동의어 사전 로드 (전기설비 진단 도메인)"""
        return {
            # 기술 용어
            "변압기": ["트랜스포머", "트랜스", "변압장치"],
            "절연저항": ["절연", "절연능력", "절연상태"],
            "DGA": ["용존가스분석", "가스분석", "절연유가스분석"],
            "절연유": ["절연오일", "트랜스오일", "절연액"],
            "진단": ["점검", "검사", "평가", "분석"],
            "기준": ["규격", "표준", "기준값", "기준치"],
            "측정": ["검측", "계측", "측정값"],
            
            # 일반 용어
            "방법": ["절차", "과정", "기법", "수단"],
            "원인": ["요인", "이유", "근거"],
            "결과": ["효과", "영향", "성과"],
            "문제": ["이상", "장애", "고장", "결함"],
            "해결": ["조치", "대응", "처리"],
            
            # 동작 용어
            "확인": ["점검", "검증", "체크"],
            "분석": ["평가", "검토"],
            "비교": ["대조", "대비", "대비분석"],
            "설명": ["해설", "안내"]
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        쿼리에서 키워드 추출 (형태소 분석 기반)
        
        Args:
            query: 쿼리 문자열
            
        Returns:
            키워드 리스트
        """
        if not query or len(query.strip()) == 0:
            return []
        
        # 형태소 분석 사용 가능한 경우
        if self.use_morphological and self.kiwi:
            try:
                return self._extract_keywords_morphological(query)
            except Exception as e:
                self.logger.warning(f"형태소 분석 실패, 정규식 방식으로 폴백: {str(e)}")
        
        # 폴백: 정규식 기반 추출
        return self._extract_keywords_regex(query)
    
    def _extract_keywords_morphological(self, query: str) -> List[str]:
        """형태소 분석 기반 키워드 추출"""
        keywords = []
        
        try:
            # 형태소 분석
            morphs = self.kiwi.analyze(query)
            
            # 명사 추출 (NNG: 일반명사, NNP: 고유명사)
            for morph, pos, start, end in morphs:
                if pos.startswith('NNG') or pos.startswith('NNP'):
                    keywords.append(morph)
                    
                    # 복합명사인 경우 분해 시도
                    if len(morph) > 2:  # 2자 이상인 경우만
                        compound_components = self._decompose_compound_word(morph, pos)
                        keywords.extend(compound_components)
            
            # 중복 제거 및 정렬
            keywords = sorted(set(keywords))
            
        except Exception as e:
            self.logger.error(f"형태소 분석 오류: {str(e)}")
            # 폴백
            return self._extract_keywords_regex(query)
        
        return keywords
    
    def _extract_keywords_regex(self, query: str) -> List[str]:
        """정규식 기반 키워드 추출 (폴백)"""
        import re
        
        # 한국어 단어 추출 (2자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', query)
        
        # 영문 단어 추출 (3자 이상)
        english_words = re.findall(r'[A-Za-z]{3,}', query)
        
        # 숫자 포함 용어 추출
        number_terms = re.findall(r'[가-힣]+[0-9]+|[0-9]+[가-힣]+', query)
        
        # 모든 키워드 합치기
        keywords = korean_words + english_words + number_terms
        
        # 중복 제거 및 정렬
        keywords = sorted(set(keywords))
        
        return keywords
    
    def _decompose_compound_word(self, word: str, pos: str) -> List[str]:
        """
        복합어 분해
        
        Args:
            word: 복합어
            pos: 품사
            
        Returns:
            분해된 구성요소 리스트
        """
        if not self.kiwi or len(word) <= 2:
            return []
        
        components = []
        try:
            # 복합어를 다시 형태소 분석하여 구성요소 추출
            morphs = self.kiwi.analyze(word)
            for morph, p, _, _ in morphs:
                if p.startswith('NN') and morph != word:  # 원본과 다른 명사 구성요소
                    components.append(morph)
        except Exception:
            pass
        
        return components
    
    def _get_synonyms_by_pos(self, term: str, pos: str) -> List[str]:
        """
        품사별 동의어 조회
        
        Args:
            term: 용어
            pos: 품사
            
        Returns:
            동의어 리스트
        """
        # 품사별 동의어 사전에서 조회
        if pos.startswith('NN') and pos in self.pos_synonym_dict:
            if term in self.pos_synonym_dict[pos]:
                return self.pos_synonym_dict[pos][term]
        
        # 폴백: 기존 동의어 사전에서 조회
        return self.synonym_dict.get(term, [])
    
    def expand_query(self, query: str, max_expansions: int = 3) -> str:
        """
        쿼리 확장 (형태소 분석 기반)
        
        Args:
            query: 원본 쿼리
            max_expansions: 최대 확장 용어 수
            
        Returns:
            확장된 쿼리
        """
        if not query or len(query.strip()) == 0:
            return query
        
        # 형태소 분석 사용 시
        if self.use_morphological and self.kiwi:
            try:
                return self._expand_query_morphological(query, max_expansions)
            except Exception as e:
                self.logger.warning(f"형태소 분석 기반 확장 실패, 기본 방식으로 폴백: {str(e)}")
        
        # 폴백: 기존 방식
        return self._expand_query_basic(query, max_expansions)
    
    def _expand_query_morphological(self, query: str, max_expansions: int) -> str:
        """형태소 분석 기반 쿼리 확장"""
        try:
            # 형태소 분석
            morphs = self.kiwi.analyze(query)
            
            # 확장된 용어 집합
            expanded_terms = set()
            original_terms = []
            
            # 품사별 키워드 추출 및 동의어 확장
            for morph, pos, start, end in morphs:
                if pos.startswith('NN'):  # 명사만 처리
                    original_terms.append(morph)
                    expanded_terms.add(morph)
                    
                    # 품사별 동의어 확장
                    synonyms = self._get_synonyms_by_pos(morph, pos)
                    if synonyms:
                        expanded_terms.update(synonyms[:max_expansions])
                    
                    # 복합어 분해 및 동의어 확장
                    if len(morph) > 2:
                        components = self._decompose_compound_word(morph, pos)
                        for comp in components:
                            expanded_terms.add(comp)
                            comp_synonyms = self._get_synonyms_by_pos(comp, pos)
                            if comp_synonyms:
                                expanded_terms.update(comp_synonyms[:max_expansions])
            
            # 확장된 쿼리 구성
            if len(expanded_terms) > len(original_terms):
                new_terms = expanded_terms - set(original_terms)
                if new_terms:
                    expanded_query = query + " " + " ".join(list(new_terms)[:max_expansions * 2])
                    
                    self.logger.debug(
                        f"쿼리 확장 (형태소 분석): '{query}' -> '{expanded_query}' "
                        f"(추가 용어: {len(new_terms)}개)"
                    )
                    return expanded_query
            
            return query
            
        except Exception as e:
            self.logger.error(f"형태소 분석 기반 확장 오류: {str(e)}")
            return self._expand_query_basic(query, max_expansions)
    
    def _expand_query_basic(self, query: str, max_expansions: int) -> str:
        """기본 쿼리 확장 (기존 방식)"""
        # 쿼리에서 키워드 추출
        keywords = self._extract_keywords_regex(query)
        
        # 각 키워드에 대한 동의어 찾기
        expanded_terms = set(keywords)
        for keyword in keywords:
            synonyms = self.synonym_dict.get(keyword, [])
            if synonyms:
                # 최대 확장 수만큼만 추가
                expanded_terms.update(synonyms[:max_expansions])
        
        # 확장된 쿼리 구성
        if len(expanded_terms) > len(keywords):
            # 원본 쿼리에 동의어 추가
            expanded_query = query
            new_terms = expanded_terms - set(keywords)
            if new_terms:
                expanded_query += " " + " ".join(list(new_terms)[:max_expansions])
            
            self.logger.debug(
                f"쿼리 확장 (기본): '{query}' -> '{expanded_query}' "
                f"(추가 용어: {len(new_terms)}개)"
            )
            return expanded_query
        else:
            return query
    
    def expand_query_with_weights(
        self,
        query: str,
        max_expansions: int = 3
    ) -> Dict[str, float]:
        """
        가중치가 있는 쿼리 확장 (형태소 분석 기반)
        
        원본 용어는 높은 가중치, 동의어는 낮은 가중치를 부여합니다.
        
        Args:
            query: 원본 쿼리
            max_expansions: 최대 확장 용어 수
            
        Returns:
            용어별 가중치 딕셔너리
        """
        term_weights = {}
        
        # 형태소 분석 사용 시
        if self.use_morphological and self.kiwi:
            try:
                return self._expand_query_with_weights_morphological(query, max_expansions)
            except Exception as e:
                self.logger.warning(f"형태소 분석 기반 가중치 확장 실패, 기본 방식으로 폴백: {str(e)}")
        
        # 폴백: 기본 방식
        keywords = self._extract_keywords_regex(query)
        
        # 원본 키워드는 가중치 1.0
        for keyword in keywords:
            term_weights[keyword] = 1.0
        
        # 동의어는 가중치 0.5
        for keyword in keywords:
            synonyms = self.synonym_dict.get(keyword, [])
            for synonym in synonyms[:max_expansions]:
                if synonym not in term_weights:
                    term_weights[synonym] = 0.5
        
        return term_weights
    
    def _expand_query_with_weights_morphological(
        self,
        query: str,
        max_expansions: int
    ) -> Dict[str, float]:
        """형태소 분석 기반 가중치 확장"""
        term_weights = {}
        
        try:
            # 형태소 분석
            morphs = self.kiwi.analyze(query)
            
            # 원본 키워드 추출 및 가중치 할당
            for morph, pos, start, end in morphs:
                if pos.startswith('NN'):  # 명사만 처리
                    # 원본 키워드: 가중치 1.0
                    term_weights[morph] = 1.0
                    
                    # 품사별 동의어 확장: 가중치 0.7
                    synonyms = self._get_synonyms_by_pos(morph, pos)
                    for synonym in synonyms[:max_expansions]:
                        if synonym not in term_weights:
                            term_weights[synonym] = 0.7
                    
                    # 복합어 분해 및 구성요소: 가중치 0.5
                    if len(morph) > 2:
                        components = self._decompose_compound_word(morph, pos)
                        for comp in components:
                            if comp not in term_weights:
                                term_weights[comp] = 0.5
                            
                            # 구성요소의 동의어: 가중치 0.3
                            comp_synonyms = self._get_synonyms_by_pos(comp, pos)
                            for comp_syn in comp_synonyms[:max_expansions]:
                                if comp_syn not in term_weights:
                                    term_weights[comp_syn] = 0.3
                    
        except Exception as e:
            self.logger.error(f"형태소 분석 기반 가중치 확장 오류: {str(e)}")
            # 폴백
            keywords = self._extract_keywords_regex(query)
            for keyword in keywords:
                term_weights[keyword] = 1.0
                synonyms = self.synonym_dict.get(keyword, [])
                for synonym in synonyms[:max_expansions]:
                    if synonym not in term_weights:
                        term_weights[synonym] = 0.5
        
        return term_weights
    
    def add_synonym(self, term: str, synonym: str, pos: Optional[str] = None):
        """
        동의어 추가
        
        Args:
            term: 원본 용어
            synonym: 동의어
            pos: 품사 (선택적, 형태소 분석 사용 시)
        """
        if term not in self.synonym_dict:
            self.synonym_dict[term] = []
        
        if synonym not in self.synonym_dict[term]:
            self.synonym_dict[term].append(synonym)
            self.logger.debug(f"동의어 추가: {term} -> {synonym}")
            
            # 품사별 동의어 사전에도 추가 (형태소 분석 사용 시)
            if self.use_morphological and self.kiwi and pos:
                if pos.startswith('NN'):
                    if pos not in self.pos_synonym_dict:
                        self.pos_synonym_dict[pos] = {}
                    if term not in self.pos_synonym_dict[pos]:
                        self.pos_synonym_dict[pos][term] = []
                    if synonym not in self.pos_synonym_dict[pos][term]:
                        self.pos_synonym_dict[pos][term].append(synonym)
    
    def load_synonyms_from_file(self, file_path: str):
        """
        파일에서 동의어 사전 로드
        
        Args:
            file_path: 동의어 사전 파일 경로 (JSON 또는 YAML)
        """
        import json
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            self.logger.warning(f"동의어 사전 파일이 없습니다: {file_path}")
            return
        
        try:
            if path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.synonym_dict.update(data)
            elif path.suffix in ['.yaml', '.yml']:
                import yaml
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    self.synonym_dict.update(data)
            
            # 품사별 동의어 사전 재구축
            if self.use_morphological:
                self._build_pos_synonym_dict()
            
            self.logger.info(f"동의어 사전 로드 완료: {file_path} ({len(self.synonym_dict)}개 항목)")
        except Exception as e:
            self.logger.error(f"동의어 사전 로드 실패: {str(e)}")
