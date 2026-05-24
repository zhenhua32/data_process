import json
import os

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from dotenv import load_dotenv

import params as params_model


def init_env(env_file: str):
    """ "
    从文本中加载并覆盖环境变量
    """
    ok = load_dotenv(dotenv_path=env_file, verbose=True, override=True)
    if not ok:
        raise Exception("load dotenv failed")


class Client:
    def __init__(self, env_file: str) -> None:
        init_env(env_file)
        self.client = AcsClient(
            os.environ["AccessKeyId"],
            os.environ["AccessKeySecret"],
            "cn-hangzhou"
        )

    def request(self, params: params_model.BaseReq):
        request = CommonRequest()

        # domain和version是固定值
        request.set_domain("alinlp.cn-hangzhou.aliyuncs.com")
        request.set_version("2020-06-29")

        request.set_action_name(params.Action)
        for key, val in params.dict().items():
            if key != "Action":
                request.add_query_param(key, val.value if isinstance(val, params_model.Enum) else val)

        response = self.client.do_action_with_exception(request)
        resp_obj = json.loads(response)
        return resp_obj
    
    def request_without_exception(self, params: params_model.BaseReq):
        try:
            result = self.request(params)
        except Exception as e:
            result = {"exception": str(e)}
        return result


if __name__ == "__main__":
    client = Client(".env")
    params = params_model.ReqGetWsChGeneral(Text="我是中国人")
    resp = client.request(params)
    print(resp)
