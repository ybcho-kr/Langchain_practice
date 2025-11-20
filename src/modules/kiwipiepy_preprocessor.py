"""
Kiwipiepy 기반 형태소 전처리기

문서 저장 및 검색 시 조사/어미를 제외한 형태소만 남기기 위한 유틸리티입니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger

try:
    from kiwipiepy import Kiwi  # type: ignore

    KIWIPIEPY_AVAILABLE = True
except ImportError:  # pragma: no cover - Kiwipiepy 미설치 환경 대비
    Kiwi = None
    KIWIPIEPY_AVAILABLE = False


class KiwipiepyPreprocessor:
    """Kiwipiepy 형태소 분석 전처리기"""

    EXCLUDE_POS_PREFIXES = ("J", "E")  # 조사(J), 어미(E)

    def __init__(
        self,
        use_kiwipiepy: bool = True,
        dictionary_path: Optional[str] = None,
    ) -> None:
        self.logger = get_logger()
        self.dictionary_path = dictionary_path
        self.user_dictionary_path: Optional[str] = None
        self.use_kiwipiepy = use_kiwipiepy and KIWIPIEPY_AVAILABLE
        self.kiwi: Optional["Kiwi"] = None

        if not self.use_kiwipiepy:
            if use_kiwipiepy:
                if not KIWIPIEPY_AVAILABLE:
                    self.logger.warning(
                        "Kiwipiepy가 설치되어 있지 않아 형태소 전처리를 비활성화합니다. "
                        "설치 방법: pip install kiwipiepy"
                    )
                else:
                    self.logger.warning(
                        "Kiwipiepy가 비활성화되었습니다 (use_kiwipiepy=False 또는 KIWIPIEPY_AVAILABLE=False)"
                    )
            return

        self._initialize_kiwi()

    # ------------------------------------------------------------------
    def _initialize_kiwi(self) -> None:
        """Kiwipiepy Kiwi 초기화"""
        if not KIWIPIEPY_AVAILABLE:
            self.logger.warning("KIWIPIEPY_AVAILABLE=False: Kiwipiepy 모듈이 import되지 않았습니다. 'pip install kiwipiepy' 실행 필요")
            return

        try:
            # Kiwi 초기화 (기본 사전 내장)
            self.kiwi = Kiwi()
            
            # 사용자 사전 경로가 제공된 경우 로드
            if self.dictionary_path:
                try:
                    # kiwipiepy는 사용자 사전을 파일로 직접 로드하는 기능이 제한적
                    # 대신 add_user_word를 사용하거나, 사전 파일을 파싱하여 추가
                    self.logger.info(f"사용자 사전 경로 제공됨: {self.dictionary_path} (kiwipiepy는 기본 사전 사용)")
                except Exception as e:
                    self.logger.warning(f"사용자 사전 로드 실패: {e}")
            
            # 초기화 테스트
            test_result = self.kiwi.analyze("테스트")
            if test_result:
                self.logger.info("Kiwipiepy 형태소 분석기를 초기화했습니다")
            else:
                raise RuntimeError("Kiwipiepy 초기화 테스트 실패")
                
        except Exception as exc:  # pragma: no cover - 환경 의존
            self.logger.error(f"Kiwipiepy 초기화 실패. 전처리를 비활성화합니다: {exc}")
            import traceback
            self.logger.error(f"Kiwipiepy 초기화 상세 오류:\n{traceback.format_exc()}")
            self.kiwi = None
            self.use_kiwipiepy = False

    # ------------------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """
        입력 텍스트를 형태소 단위로 분해하고 조사/어미를 제외한 형태소만 공백으로 결합합니다.
        전처리에 실패하면 원본 텍스트를 반환합니다.
        """
        if not self.use_kiwipiepy or not self.kiwi:
            return text

        stripped = (text or "").strip()
        if not stripped:
            return stripped

        # 입력 텍스트 로깅 (INFO 레벨로 변경하여 항상 출력)
        input_preview = stripped[:100] + ('...' if len(stripped) > 100 else '')
        self.logger.info(f"[Kiwipiepy 전처리] 입력 텍스트: {input_preview}")
        
        try:
            # 입력 텍스트 타입 및 인코딩 확인
            if not isinstance(stripped, str):
                self.logger.error(f"[Kiwipiepy 전처리] 입력이 문자열이 아닙니다: {type(stripped)}")
                return text
            
            # kiwipiepy.analyze()는 List[Tuple[List[Token], float]]를 반환
            # 각 튜플은 (토큰 리스트, 분석 점수) 형식
            # Token 객체는 form, tag, start, len 속성을 가짐
            results = self.kiwi.analyze(stripped)
        except Exception as exc:  # pragma: no cover - 환경 의존
            import traceback
            self.logger.error(f"[Kiwipiepy 전처리] 분석 실패. 원본 텍스트를 사용합니다: {exc}")
            self.logger.error(f"[Kiwipiepy 전처리] 상세 오류:\n{traceback.format_exc()}")
            return text

        # 분석 결과 로깅
        if results and len(results) > 0:
            token_list, score = results[0]  # (List[Token], float)
            self.logger.info(f"[Kiwipiepy 전처리] 분석 결과 (점수: {score:.2f}): {len(token_list)}개 토큰")
            
            # 각 토큰 상세 정보 로깅 (처음 10개만)
            token_details = []
            for i, token in enumerate(token_list[:10]):
                token_details.append(f"{token.form}/{token.tag}")
            if len(token_list) > 10:
                token_details.append(f"... 외 {len(token_list) - 10}개")
            self.logger.info(f"[Kiwipiepy 전처리] 토큰 상세: {' '.join(token_details)}")
        else:
            self.logger.warning(f"[Kiwipiepy 전처리] 분석 결과가 비어있습니다: {stripped[:50]}")

        tokens: List[str] = []
        current_compound: List[str] = []  # 복합명사 구성 중인 형태소들
        
        # 명사/고유명사는 각각 개별 토큰으로 추가 (공백으로 분리)
        # 단, 연속된 고유명사+일반명사는 복합명사로 간주하여 결합
        # 품사 태그: NNG(일반명사), NNP(고유명사), NNB(의존명사), NR(수사), NP(대명사)
        NOUN_POS = ("NNG", "NNP", "NNB", "NR", "NP")
        
        # results는 (토큰 리스트, 점수) 튜플의 리스트
        # 첫 번째 결과(가장 높은 점수)를 사용
        if results and len(results) > 0:
            token_list, _ = results[0]  # (List[Token], float)
            
            excluded_count = 0
            noun_count = 0
            verb_adj_count = 0
            prev_pos = None  # 이전 토큰의 품사
            
            for i, token in enumerate(token_list):
                morph = token.form  # Token 객체의 form 속성
                pos = token.tag     # Token 객체의 tag 속성
                
                if not morph or not morph.strip() or len(morph.strip()) == 0:
                    excluded_count += 1
                    continue
                
                morph = morph.strip()
                
                # 조사/어미 제외
                if self._should_exclude_pos(pos):
                    excluded_count += 1
                    # 조사/어미가 나오면 복합명사 종료
                    if current_compound:
                        tokens.append("".join(current_compound))
                        noun_count += 1
                        current_compound = []
                    prev_pos = None
                    continue
                
                # 명사류 처리
                if pos.startswith(NOUN_POS):
                    # 이전 토큰도 명사인 경우
                    if prev_pos and prev_pos.startswith(NOUN_POS):
                        # 이전 명사가 짧고(1-2글자) 현재 명사도 짧은 경우 복합명사로 간주
                        # 예: "유중/NNP 가스/NNG" → "유중가스" (각각 2글자)
                        prev_morph = current_compound[-1] if current_compound else ""
                        if len(prev_morph) <= 2 and len(morph) <= 2:
                            # 복합명사로 결합
                            current_compound.append(morph)
                        else:
                            # 이전 명사 종료하고 현재 명사는 새로 시작
                            if current_compound:
                                tokens.append("".join(current_compound))
                                noun_count += 1
                                current_compound = []
                            # 현재 명사는 새로 시작
                            tokens.append(morph)
                            noun_count += 1
                            current_compound = []
                    else:
                        # 첫 명사이거나 이전이 명사가 아닌 경우
                        current_compound.append(morph)
                    prev_pos = pos
                else:
                    # 명사가 아닌 경우, 복합명사 종료 후 추가
                    if current_compound:
                        tokens.append("".join(current_compound))
                        noun_count += 1
                        current_compound = []
                    # 동사/형용사 어간 등은 그대로 추가
                    tokens.append(morph)
                    verb_adj_count += 1
                    prev_pos = None
            
            # 마지막 복합명사 처리
            if current_compound:
                tokens.append("".join(current_compound))
                noun_count += 1
            
            # 필터링 결과 로깅 (품사별 통계 포함)
            self.logger.info(
                f"[Kiwipiepy 전처리] 필터링 결과: {len(tokens)}개 단어 포함 "
                f"(명사: {noun_count}개, 동사/형용사: {verb_adj_count}개), {excluded_count}개 제외"
            )

        if not tokens:
            self.logger.info(f"[Kiwipiepy 전처리] 필터링 후 토큰이 없어 원본 텍스트 반환")
            return stripped

        # 공백으로 결합 (각 명사가 개별 토큰으로 분리됨)
        result_text = " ".join(tokens)
        result_preview = result_text[:100] + ('...' if len(result_text) > 100 else '')
        
        # 전처리 전후 비교 로깅
        input_length = len(stripped)
        output_length = len(result_text)
        token_count = len(tokens)
        
        self.logger.info(f"[Kiwipiepy 전처리] 최종 결과: {result_preview}")
        self.logger.info(
            f"[Kiwipiepy 전처리] 전처리 통계: "
            f"입력 길이 {input_length}자 → 출력 길이 {output_length}자, "
            f"토큰 수 {token_count}개"
        )
        
        return result_text

    # ------------------------------------------------------------------
    def load_custom_dictionary(self, dictionary_path: str) -> bool:
        """
        사용자 정의 사전을 로딩합니다. (추후 확장용)

        Args:
            dictionary_path: 사용자 사전 파일 경로 (kiwipiepy는 add_user_word 방식 사용)

        Returns:
            bool: 로딩 성공 여부
        """
        if not self.use_kiwipiepy or not self.kiwi:
            self.logger.warning("Kiwipiepy가 비활성화되어 사용자 사전을 로드할 수 없습니다.")
            return False

        dic_path = Path(dictionary_path)
        if not dic_path.exists():
            self.logger.error(f"지정한 사용자 사전을 찾을 수 없습니다: {dic_path}")
            return False

        try:
            # kiwipiepy는 파일 기반 사용자 사전을 직접 지원하지 않음
            # 대신 파일을 읽어서 add_user_word로 추가해야 함
            # 간단한 구현: 한 줄에 하나의 단어 형식 가정
            with open(dic_path, 'r', encoding='utf-8') as f:
                word_count = 0
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 형식: "단어\t품사" 또는 "단어"
                        parts = line.split('\t')
                        word = parts[0].strip()
                        tag = parts[1].strip() if len(parts) > 1 else 'NNP'  # 기본값: 고유명사
                        if word:
                            self.kiwi.add_user_word(word, tag)
                            word_count += 1
            
            self.user_dictionary_path = str(dic_path)
            self.logger.info(f"Kiwipiepy 사용자 사전을 로드했습니다: {dic_path} ({word_count}개 단어)")
            return True
        except Exception as e:
            self.logger.error(f"사용자 사전 로드 실패: {e}")
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def _should_exclude_pos(pos: str) -> bool:
        """제외할 품사인지 여부를 판단"""
        if not pos:
            return True  # 품사가 비어 있으면 제외
        return pos.startswith(KiwipiepyPreprocessor.EXCLUDE_POS_PREFIXES)

