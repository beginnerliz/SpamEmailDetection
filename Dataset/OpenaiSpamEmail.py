# Creator Cui Liz
# Time 03/07/2024 17:58

import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
import time
import random
from tqdm import tqdm
import pandas as pd
import threading
from rich.progress import Progress, track

from EmailDataProcess import SpamIsHamDataset

dataset = SpamIsHamDataset()
dataset.load_data()
contents_list = dataset.content_list
labels_list = dataset.label_list

# 加载 .env 文件中的环境变量
load_dotenv()

RateLimitError = getattr(openai, 'RateLimitError', None)
OpenAIError = getattr(openai, 'OpenAIError', None)

if RateLimitError is None or OpenAIError is None:
    print("Error classes not found in openai.error")

# 获取 API 密钥
api_key = os.getenv('OPENAI_API_KEY')

# 检查是否成功获取到 API 密钥
if not api_key:
    raise ValueError("API key not found. Make sure .env file is set up correctly and contains the OPENAI_API_KEY.")

client = OpenAI(api_key=api_key)


def make_request(command: str):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an email writer."},
                {"role": "user", "content": command}
            ]
        )
        return response
    except RateLimitError as e:
        print(f"Rate limit error: {e}")
    except OpenAIError as e:
        print(f"OpenAI error: {e}")


def getTopics():
    n_rounds = 10
    topics_per_round = 100

    all_topics = []

    for round in tqdm(range(n_rounds), desc="Generating Topics"):
        response = make_request(
            f"Give me a list of {topics_per_round} spam email topics, write them in csv format.")
        response_content = response.choices[0].message.content

        all_topics.extend(response_content.split("\n"))
        time.sleep(0.5)

    dataframe = pd.DataFrame(all_topics, columns=["Topics"])
    dataframe.to_csv("topics.csv", index=False)


def getSpamEmails():
    from Dataset.Prompts.command_specs import command_specs
    topics_df = pd.read_csv("Prompts/Topics.csv")
    topics = topics_df["Subject"].tolist()

    num_emails = 5
    additional_dataset = []

    for i in tqdm(range(num_emails), desc="ChatGPT is busy 🐢："):
        topic = random.choice(topics)
        n_words = random.randint(50, 500)
        spec = random.choice(command_specs)
        try:
            command = (f"Write one short spam email with around {n_words} words, the topic is: {topic}."
                       f"The email starts with '<start>' and end with '<end>'. "
                       f"{spec}")

            response = make_request(command)
            response_content = response.choices[0].message.content

            start = response_content.find("<start>") + 7
            end = response_content.rfind("<end>")
            pure_content = response_content[start:end].strip(".,\n ")

            additional_dataset.append(("1", pure_content))
            time.sleep(0.5)
        except Exception as e:
            print("Error Occurred, skipped.", e)

    df = pd.DataFrame(additional_dataset, columns=["label", "text"])
    df.to_csv("GPT_Gen.csv", index=False)


def openai_revised(task_id, pbar):
    additional_dataset = []

    for i in range(emails_per_thread):
        try:
            chosen_id = random.randint(0, len(labels_list))
            command = (f"According to the raw email:\n {contents_list[chosen_id]}, "
                       f"please refine the wrong word spelling and grammar."
                       f"Only need then email body."
                       )
            response_rephrase = make_request(command)
            response_content = response_rephrase.choices[0].message.content
            additional_dataset.append((labels_list[chosen_id], response_content))
            time.sleep(0.8)
        except Exception as e:
            print("Error Occurred, skipped.", e)
        progress.update(task_id, advance=1)

    df = pd.DataFrame(additional_dataset, columns=["label", "text"])
    df.to_csv(f"GPT_Rev_part_{task_id+20}.csv", index=False)


if __name__ == '__main__':
    # getSpamEmails()
    progress = Progress()
    progress.start()

    n_emails = 10000
    n_threads = 5
    emails_per_thread = n_emails // n_threads

    threads = []

    for i in range(n_threads):
        task_id = progress.add_task(f"Task {i+1} 🦸", total=emails_per_thread)
        thread = threading.Thread(target=openai_revised, args=(task_id, progress))
        threads.append(thread)

    for thread in threads:
        thread.start()
        time.sleep(0.2)

    for thread in threads:
        thread.join()

    progress.stop()

    print("FINISHED.")
    # label, openai_spam = openai_revised(labels_list, contents_list)
    # print(label)
    # print(openai_spam)


