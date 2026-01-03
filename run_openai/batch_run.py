import os
import json

from modelscope_model import create_image_generations, get_task_result


def load_prompt_file(file_path: str = "../data/opennana/prompts.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        prompts_info = json.load(file)
    
    prompt_list = []
    for item in prompts_info["items"]:
        prompt_list.append(item["prompts"][-1])  # 选择 -1, 后面的大概率为中文版本的
    return prompt_list


def run_batch_image_generation(prompt_list, output_dir: str = "./output_images"):
    model_names = [
        "Qwen/Qwen-Image-2512",
        "Tongyi-MAI/Z-Image-Turbo",
    ]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, prompt in enumerate(prompt_list):
        for model in model_names:
            # 一天运行不完, 所以先判断下是否已经生成过
            name = model.split("/")[-1]
            output_path = os.path.join(output_dir, f"result_{idx + 1}_{name}.jpg")
            if os.path.exists(output_path):
                print(f"Image for prompt {idx + 1} with model {name} already exists, skipping.")
                continue

            print(f"Generating image for prompt {idx + 1}/{len(prompt_list)}: {prompt}")
            try:
                task_id, limit_info = create_image_generations(prompt, model=model)
            except Exception as e:
                print(f"Error creating image generation task: {e}")
                continue
            print("Task ID:", task_id)
            print("Limit Info:", limit_info)
            # 当次数用完时, 会抛出异常, 这里不做处理, 直接让程序停止
            if limit_info["modelscope_ratelimit_model_requests_remaining"] <= 20:
                print("Model request limit reached, stopping further requests.")
                return

            image, result_data = get_task_result(task_id)
            if image:
                name = model.split("/")[-1]
                image.save(os.path.join(output_dir, f"result_{idx + 1}_{name}.jpg"))
                print(f"Image saved as result_{idx + 1}_{name}.jpg")
            else:
                print(f"Failed to generate image for prompt {idx + 1}, info: {result_data}")


if __name__ == "__main__":
    prompts = load_prompt_file()
    run_batch_image_generation(prompts)
