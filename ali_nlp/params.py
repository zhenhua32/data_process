from enum import Enum
from pydantic import BaseModel, constr, Field


class OutTypeEnum(str, Enum):
    Zero = "0"
    One = "1"
    Two = "2"


class BaseReq(BaseModel):
    """
    基础请求参数类
    """

    Action: str
    ServiceCode: str = Field(default="alinlp", const=True)


class ReqGetWsChGeneral(BaseReq):
    """
    中文分词(基础版-通用领域)
    """

    Action: str = Field(default="GetWsChGeneral", const=True)
    Text: constr(min_length=1, max_length=1024)
    TokenizerId: str = Field(default="GENERAL_CHN", const=True)
    OutType: OutTypeEnum = Field(default=OutTypeEnum.One)
