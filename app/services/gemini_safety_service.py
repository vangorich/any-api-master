# -*- coding: utf-8 -*-
from typing import List, Dict, Any

# 定义 Gemini API 的所有已知危害类别
# 参考: https://cloud.google.com/vertex-ai/docs/generative-ai/multimodal/configure-safety-attributes
ALL_HARM_CATEGORIES = [
    # PaLM Legacy Categories
    "HARM_CATEGORY_DEROGATORY",         # 负面或有害评论
    "HARM_CATEGORY_TOXICITY",           # 粗鲁、不敬或亵渎内容
    "HARM_CATEGORY_VIOLENCE",           # 描述暴力场景
    "HARM_CATEGORY_SEXUAL",             # 包含性行为或淫秽内容
    "HARM_CATEGORY_MEDICAL",            # 宣传未经证实的医疗建议
    "HARM_CATEGORY_DANGEROUS",          # 宣扬有害行为的危险内容
    # Gemini Categories
    "HARM_CATEGORY_HARASSMENT",         # 骚扰内容
    "HARM_CATEGORY_HATE_SPEECH",        # 仇恨言论
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",  # 露骨色情内容
    "HARM_CATEGORY_DANGEROUS_CONTENT",  # 危险内容
]

# 定义安全阈值
# 参考: https://ai.google.dev/docs/safety_setting_gemini
class HarmBlockThreshold:
    """安全设置的阈值"""
    HARM_BLOCK_THRESHOLD_UNSPECIFIED = "HARM_BLOCK_THRESHOLD_UNSPECIFIED"   # 阈值未指定。
    BLOCK_LOW_AND_ABOVE = "BLOCK_LOW_AND_ABOVE"                             # 内容带有“可忽略”字样，允许使用。
    BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"                       # 内容风险为“可忽略”和“低”时，将允许投放广告。
    BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"                                     # 系统将允许发布风险为“可忽略”“低”和“中”的内容。
    BLOCK_NONE = "BLOCK_NONE"                                               # 允许所有内容。
    OFF = "OFF"                                                             # 关闭安全过滤条件。

class GeminiSafetyService:
    def get_default_safety_settings(self) -> List[Dict[str, Any]]:
        """
        生成默认的安全设置列表，将所有类别的阈值设置为 OFF。
        """
        return [
            {
                "category": category,
                "threshold": HarmBlockThreshold.OFF,
            }
            for category in ALL_HARM_CATEGORIES
        ]

    def add_safety_settings_to_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        将默认的安全设置添加到请求体中。
        如果请求体中已经存在 safetySettings，则不进行任何操作。
        """
        if "safetySettings" not in payload:
            payload["safetySettings"] = self.get_default_safety_settings()
        return payload

# 创建一个单例
gemini_safety_service = GeminiSafetyService()