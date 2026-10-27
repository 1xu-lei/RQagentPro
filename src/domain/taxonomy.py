"""
标签体系中心模块 — 系统唯一真相源

数据从 data/domains.yaml 与 data/regulations.yaml 加载（不再硬编码在代码里）：
  - data/domains.yaml     12 个一级域 / 65 个二级标签 / 旧域名别名映射
  - data/regulations.yaml 法规依据表（REGULATION_MAP）

所有意图分类、安全检测、编码输出均以此模块为准。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


# ============================================================
# 数据文件路径
# ============================================================

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
DOMAINS_YAML = _DATA_DIR / "domains.yaml"
REGULATIONS_YAML = _DATA_DIR / "regulations.yaml"


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SubLabel:
    """二级标签"""
    code: str           # 如 "BZ-01"
    name: str           # 如 "居民新装报装"
    examples: str = ""  # 典型问句
    regulation: str = ""  # 关联法规编码
    safety_sensitive: bool = False


@dataclass
class Domain:
    """一级业务域"""
    code: str           # 如 "BZ"
    name: str           # 如 "报装开户"
    sub_labels: List[SubLabel] = field(default_factory=list)

    @property
    def safety_labels(self) -> List[SubLabel]:
        return [sl for sl in self.sub_labels if sl.safety_sensitive]

    @property
    def label_count(self) -> int:
        return len(self.sub_labels)


# ============================================================
# 从 YAML 加载（12 个一级域 + 65 个二级标签 + 法规映射）
# ============================================================

def _load_domains() -> List[Domain]:
    """从 data/domains.yaml 加载标签体系"""
    with open(DOMAINS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    domains: List[Domain] = []
    for d in data.get("domains", []):
        sub_labels = [
            SubLabel(
                code=sl["code"],
                name=sl["name"],
                examples=sl.get("examples", ""),
                regulation=sl.get("regulation", ""),
                safety_sensitive=bool(sl.get("safety_sensitive", False)),
            )
            for sl in d.get("sub_labels", [])
        ]
        domains.append(Domain(d["code"], d["name"], sub_labels))
    return domains


def _load_aliases() -> Dict[str, str]:
    """加载旧域名 → 规范域名映射；规范名自映射由代码生成"""
    with open(DOMAINS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    aliases: Dict[str, str] = {d.name: d.name for d in DOMAINS}
    aliases.update(data.get("aliases", {}))
    return aliases


def _load_regulation_map() -> Dict[str, Dict]:
    """从 data/regulations.yaml 加载法规依据表"""
    with open(REGULATIONS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


DOMAINS: List[Domain] = _load_domains()
REGULATION_MAP: Dict[str, Dict] = _load_regulation_map()
KB_ALIAS_TO_STANDARD: Dict[str, str] = _load_aliases()


# ============================================================
# 索引结构（O(1) 查询）
# ============================================================

# 编码 → Domain
DOMAIN_BY_CODE: Dict[str, Domain] = {d.code: d for d in DOMAINS}

# 域名 → Domain
DOMAIN_BY_NAME: Dict[str, Domain] = {d.name: d for d in DOMAINS}

# 编码 → SubLabel
SUB_LABEL_BY_CODE: Dict[str, SubLabel] = {}
for d in DOMAINS:
    for sl in d.sub_labels:
        SUB_LABEL_BY_CODE[sl.code] = sl

# 所有安全敏感标签编码集合
SAFETY_LABEL_CODES: Set[str] = {sl.code for sl in SUB_LABEL_BY_CODE.values() if sl.safety_sensitive}

# 域名 → 编码映射（code_map 的规范版本）
NAME_TO_CODE: Dict[str, str] = {d.name: d.code for d in DOMAINS}


def normalize_domain_name(raw_name: str) -> str:
    """
    将任意域名（来自 KB、意图分类、旧 code_map）标准化为规范域名

    Args:
        raw_name: 原始域名

    Returns:
        规范域名；未匹配时返回 "增值与其他"
    """
    return KB_ALIAS_TO_STANDARD.get(raw_name, "增值与其他")


def get_domain_code(domain_name: str) -> str:
    """根据域名（规范名或旧名）返回规范编码"""
    std_name = normalize_domain_name(domain_name)
    return NAME_TO_CODE.get(std_name, "QT")


def get_safety_info(sub_label_code: str) -> Optional[SubLabel]:
    """获取安全标签信息；非安全标签返回 None"""
    sl = SUB_LABEL_BY_CODE.get(sub_label_code)
    if sl and sl.safety_sensitive:
        return sl
    return None


def is_safety_label(code: str) -> bool:
    """判断标签编码是否为安全敏感标签"""
    return code in SAFETY_LABEL_CODES


def list_safety_labels() -> List[SubLabel]:
    """列出所有安全敏感标签"""
    return [sl for sl in SUB_LABEL_BY_CODE.values() if sl.safety_sensitive]


def get_all_domain_names() -> List[str]:
    """返回 12 个规范域名列表"""
    return [d.name for d in DOMAINS]


def get_all_domain_codes() -> Dict[str, str]:
    """返回 {域名: 编码} 完整映射"""
    return dict(NAME_TO_CODE)


# ============================================================
# 法规映射
# ============================================================

def get_regulation_info(code: str) -> Optional[Dict]:
    """根据法规编码获取法规信息"""
    return REGULATION_MAP.get(code)


# ============================================================
# 便捷函数
# ============================================================

def print_taxonomy_summary():
    """打印标签体系摘要"""
    total_labels = sum(d.label_count for d in DOMAINS)
    safety_count = len(SAFETY_LABEL_CODES)
    print(f"标签体系: {len(DOMAINS)} 个一级域, {total_labels} 个二级标签")
    print(f"安全敏感标签: {safety_count} 个 ★")
    for d in DOMAINS:
        safety = [sl.code for sl in d.sub_labels if sl.safety_sensitive]
        safety_str = f"  ★{len(safety)}个: {', '.join(safety)}" if safety else ""
        print(f"  {d.code} {d.name}: {d.label_count}个标签{safety_str}")
