# ============================================================
#  first-hello 网页版后端
#  接口:
#    GET  /            返回页面
#    POST /generate    收 {me, jd} → 流式返回生成文本
#    POST /parse_pdf   收 PDF 文件 → 返回抽取的文字
# ============================================================

import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from openai import OpenAI
from pypdf import PdfReader

app = Flask(__name__)

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    timeout=60,
)

system_prompt = """你是一位求职沟通顾问。根据候选人的真实素材和目标岗位JD,生成:
1.【打招呼语】50字以内,用于BOSS直聘首次沟通。
2.【自我介绍】250~300字,用于官网网申的"自我评价/个人简介"栏(常见字数限制200~500字,按300字为目标写,确保删减两句后仍在200字内成立)。

打招呼语写作要求:
- 先在心里找出素材与JD之间"最独特的一条匹配线":不是候选人最大的头衔或最新的经历,而是与这个岗位的具体研究内容/业务最直接相关的那条经历(哪怕它只是一个课程项目)。打招呼语必须围绕它展开,并说清"我的经历和你们做的事有什么关系",不许只罗列经历名词。
- 它是即时通讯里发给真人的消息,必须像自然的口语开场,而不是书面陈述或简历摘要。技术细节点到为止(不必罗列每个算法名),把字数留给"关系"和"姿态"。
- 第一句必须同时完成两件事:自然报出身份 + 立刻带出最独特匹配点,一口气说完,如"您好,我是XX大学XX专业XX方向学生,做过XX项目,和贵司XX方向很贴近"。身份部分的规则:学校和专业名称必须完整出现(如"悉尼大学电气工程"),这是固定项;素材中若有专业方向/细分(如"人工智能方向"),默认必须一并带出,仅当该方向与岗位明显无关时才省略——尤其当岗位属于AI/算法/大模型类而素材有AI方向时,漏报方向属于严重错误;学位类型、"荣誉"字样、括号注释一律不写。禁止报完身份就断句另起,禁止把学校拖到第二句之后,不要提及具体年级(如"大二""大三");如素材提供了毕业时间或可实习时长,优先用它们传递阶段信息。
- 收尾必须是礼貌地请求沟通机会(如"不知是否方便进一步沟通"),而不是发起对等评估(禁止"聊聊匹配度""看看是否合适"这类表述)——评估是对方的事,候选人表达的是意愿和尊重。
- 到岗时间/实习时长:仅当素材中明确提供、且岗位为实习岗时,在结尾简洁给出。素材中没有的,一个字都不要提,严禁"近期可到岗""时间可沟通"等含糊占位。
- 语气自然、不卑不亢,禁止感叹号堆砌和"非常""十分"等空洞热情词。

自我介绍写作要求:
- 场景是网申表单,比打招呼语正式。
- 必须按逻辑分段,段与段之间空一行,共3~4段:第一段,身份背景(学校、专业、年级、方向),一两句;第二段,与该岗位最相关的1~2段经历,写清"做了什么+怎么做的+关注了什么",技术细节在这里展开;第三段,次相关的支撑经历(实习、其他项目)简要带过;末段,一句话的能力概括与求职意愿收尾。
- 经历的选择和排序始终以"与该岗位的相关度"为准,禁止平铺全部履历,禁止只堆名词。
- 禁止任何性格与软素质的自我描述("性格开朗""好奇心强""细致耐心"等一律不写),全文只写经历、方法和能力事实。

铁律:只能使用素材中真实存在的经历和技能,严禁编造、夸大或推测素材中没有的内容。如果素材与JD的匹配是能力迁移型(没有直接对口经历),如实以迁移角度表达,不要写"我在XX方面有提升空间/经验不足"这类自我扣分句式,也不要假装熟悉素材中不存在的领域。"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    me = data.get("me", "")
    jd = data.get("jd", "")
    if not me or not jd:
        return jsonify({"error": "me 和 jd 都不能为空"}), 400

    def stream():
        try:
            response = client.chat.completions.create(
                model=os.environ["LLM_MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"【候选人素材】\n{me}\n\n【目标岗位JD】\n{jd}"},
                ],
                stream=True,                   # 灵魂参数:让模型边生成边发
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta                # 收到一块,立刻转发一块
        except Exception as e:
            # 响应头已按 200 发出,错误无法再改状态码,只能作为正文补发一段。
            # 以"⚠️ 生成中断"开头,前端据此识别为错误:显示出来、且不存入历史。
            yield f"\n\n⚠️ 生成中断:{type(e).__name__}: {e}"

    return Response(stream_with_context(stream()), mimetype="text/plain")


@app.route("/parse_pdf", methods=["POST"])
def parse_pdf():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "请上传 PDF 文件"}), 400

    reader = PdfReader(f)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()

    if not text:
        return jsonify({"error": "没能从这份PDF里抽出文字(可能是扫描图片版)"}), 400
    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(debug=True)