# -*- coding: utf-8 -*-
# src_common/labels.py

# EMNIST Balanced (47 classes) → 사용자 제공 매핑
EMNIST_BALANCED_ID2CHAR = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4',
    5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E',
    15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
    20:'K', 21:'L', 22:'M', 23:'N', 24:'O',
    25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
    30:'U', 31:'V', 32:'W', 33:'X', 34:'Y',
    35:'Z', 36:'a', 37:'b', 38:'d', 39:'e',
    40:'f', 41:'g', 42:'h', 43:'n', 44:'q',
    45:'r', 46:'t'
}

def emnist_char(cid: int) -> str:
    """정수 클래스 → 문자. 매핑 밖이면 숫자 그대로 반환."""
    return EMNIST_BALANCED_ID2CHAR.get(int(cid), str(int(cid)))

def emnist_tag(cid: int) -> str:
    """파일명/제목에 쓰기 좋은 태그 '##_<char>' 형태."""
    c = int(cid)
    return f"{c:02d}_{emnist_char(c)}"
