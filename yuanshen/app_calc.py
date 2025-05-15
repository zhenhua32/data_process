import gradio as gr
import matplotlib.pyplot as plt

from core import calculate_damage

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置全局字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示


def update_damage(
    基础攻击力: float,
    面板攻击力: float,
    额外攻击力加成百分比: float,
    额外攻击力: float,
    暴击率: float,
    暴伤: float,
    元素精通: float,
    反应系数: float,
    反应加成系数: float,
    元素伤害加成: float,
    伤害加成: float,
    技能倍率: float,
    角色等级: int,
    怪物等级: int,
    减防系数: float,
    无视系数: float,
    抗性: float,
    独立乘区: float,
):
    # 处理百分比
    args = [
        基础攻击力,
        面板攻击力,
        额外攻击力加成百分比 / 100,
        额外攻击力,
        暴击率 / 100,
        暴伤 / 100,
        元素精通,
        反应系数,
        反应加成系数,
        元素伤害加成 / 100,
        伤害加成 / 100,
        技能倍率 / 100,
        角色等级,
        怪物等级,
        减防系数 / 100,
        无视系数 / 100,
        抗性 / 100,
        独立乘区 / 100,
    ]

    暴击, 非暴击, 实际 = calculate_damage(*args)

    # 生成柱状图
    fig, ax = plt.subplots()
    bars = ax.bar(["期望伤害", "非暴击", "暴击"], [实际, 非暴击, 暴击])
    ax.bar_label(bars)
    plt.close()

    # 返回多类型输出
    return (f"{实际:,.0f}", f"非暴击: {非暴击:,.0f} | 暴击: {暴击:,.0f}", fig)


with gr.Blocks(title="火神伤害计算器") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # 输入组件
            基础攻击力 = gr.Number(868, label="基础攻击力(白值)")
            面板攻击力 = gr.Number(2180, label="基础攻击力(白值+绿值)")
            额外攻击力加成百分比 = gr.Slider(minimum=0, maximum=500, value=0, step=1, label="额外攻击力加成百分比%")
            额外攻击力 = gr.Number(0, label="额外攻击力", info="班尼特大招1110")
            暴击率 = gr.Slider(0, 100, 100, step=0.1, label="暴击率%")
            暴伤 = gr.Slider(50, 300, 231.4, step=0.1, label="暴伤%")
            元素精通 = gr.Slider(0, 2000, 0, step=1, label="元素精通", info="算纯火伤时设为0")
            反应系数 = gr.Dropdown([1, 1.5, 2], value=1, label="反应系数", info="1.5: 蒸发, 2: 融化")
            反应加成系数 = gr.Number(0, label="反应加成系数")
            元素伤害加成 = gr.Slider(0, 200, 46.6, step=0.1, label="元素伤害加成%")
            伤害加成 = gr.Slider(0, 200, 50, step=0.1, label="伤害加成%", info="黑曜套15螭骨剑精一30")
            技能倍率 = gr.Slider(100, 2000, 230.4, label="技能倍率%", info="火神战技230.4, 大招1380.6")
            角色等级 = gr.Slider(1, 90, 90, step=1, label="角色等级")
            怪物等级 = gr.Slider(1, 120, 95, step=1, label="怪物等级")
            减防系数 = gr.Slider(0, 100, 0, step=0.1, label="减防系数%")
            无视系数 = gr.Slider(0, 100, 0, step=0.1, label="无视系数%")
            抗性 = gr.Slider(-100, 100, 10, step=1, label="抗性%")
            独立乘区 = gr.Slider(0, 100, 0, step=1, label="独立乘区%", info="")

        with gr.Column(scale=2):
            # 输出组件
            期望伤害 = gr.Label(label="期望伤害")
            伤害对比 = gr.Textbox(label="伤害区间")
            图表 = gr.Plot(label="伤害分布可视化")

    # 动态绑定
    inputs = [
        基础攻击力,
        面板攻击力,
        额外攻击力加成百分比,
        额外攻击力,
        暴击率,
        暴伤,
        元素精通,
        反应系数,
        反应加成系数,
        元素伤害加成,
        伤害加成,
        技能倍率,
        角色等级,
        怪物等级,
        减防系数,
        无视系数,
        抗性,
        独立乘区,
    ]
    for component in inputs:
        component.change(fn=update_damage, inputs=inputs, outputs=[期望伤害, 伤害对比, 图表])


demo.launch()
