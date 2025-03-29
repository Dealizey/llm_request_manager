from database import ConversationDB
from pprint import pprint
import tiktoken


price_USD = {
    "o1": (15, 60),
    "claude-3-7-sonnet-20250219": (3, 15),
    "o3-mini": (1.1, 4.4),
    "gemini-2.0-flash-thinking-exp-01-21": (0, 0),
}
price_CNY = {
    "ark-deepseek-r1-250120": (4, 16),
}
exchange_rate = 7.2

price_USD_converted = {key: (value[0] * exchange_rate, value[1] * exchange_rate) 
                       for key, value in price_USD.items()}

price = {**price_USD_converted, **price_CNY}

def main():
    prompt = r"""现在是2025年1月1日00:00，一亿秒之后是什么时候？"""
    encoder = tiktoken.get_encoding("cl100k_base")
    input_tokens = len(encoder.encode(prompt))
    print(f"prompt: {prompt}")

    db = ConversationDB("math.db")
    result = db.search_conversations(prompt)

    for each in result:
        model_name = each['model_name']
        total_output_tokens = each['output_tokens']
        total_time = each["execution_time"]
        output_tokens = len(encoder.encode(each['response']))
        if 'reasoning_tokens_info' in each:
            reasoning_tokens = each["reasoning_tokens_info"]["reasoning_tokens"]
        else:
            reasoning_tokens = total_output_tokens - output_tokens
        reasoning_time = total_time / total_output_tokens * reasoning_tokens
        fee = input_tokens / 1e6 * price[model_name][0] + total_output_tokens / 1e6 * price[model_name][1]

        print()
        print(f"model: {model_name}")
        print(f"{total_output_tokens=}")
        if reasoning_tokens > 15:
            print(f"{reasoning_tokens=}")
            print(f"{reasoning_time=}")
        print(f"{total_time=}")
        print(f"{fee=} CNY")
        # print(each['response'])
        print()

if __name__ == "__main__":
    main()