import argparse
import time
import uuid


def process_query(
    client, query, bot_id, iteration, operation_type, verbose, max_attempts
):
    try:
        status_msg = (
            f"\t-> Executing {operation_type}. Status: {iteration}/{max_attempts}"
        )
        print(status_msg, end="\r")

        result = client.user_message(bot_id, query)
        if verbose:
            print(result)

        for item in result:
            if (
                item.function_call
                and operation_type == item.function_call.name
                and has_bot_reply(result)
            ):
                return True, item.function_call.name

        return False, "Function not executed."
    except LLMJSONParsingError as e:
        print(f"JSON parsing failed: {e}")
        return False, "Invalid JSON response from LLM."
    except Exception as e:
        print(f"Unexpected runtime error: {e}")
        return False, "Runtime error occurred."


def has_bot_reply(result):
    for item in result:
        if item.assistant_message:
            return True
    return False


def evaluate(verbose):
    client = Client()
    print(
        "\nTest suite may take up to 30 minutes depending on system performance. New test agents will be created for each run.\n"
    )
    config = LABOConfig.load()
    print(f"version = {config.labo_version}")

    total_points, total_tokens, runtime = 0, 0, 0

    for operation_type, query in constants.TEST_QUERIES.items():
        points = 0
        start_time = time.time()
        test_id = uuid.uuid4()

        for i in range(constants.ITERATIONS):
            bot = client.create_agent(
                f"tester_{test_id}_bot_{i}",
                get_persona_text(constants.PROFILE),
                get_human_text(constants.USER_TYPE),
            )
            bot_id = bot.id
            success, status = process_query(
                client, query, bot_id, i, operation_type, verbose, constants.ITERATIONS
            )

            if verbose:
                print(f"\t{status}")

            if success:
                points += 1

        duration = time.time() - start_time
        print(
            f"Result for {operation_type}: {points}/{constants.ITERATIONS}, duration: {duration:.2f} seconds"
        )

        runtime += int(duration)
        total_points += points

    print(f"\nMEMGPT VERSION: {config.labo_version}")
    print(f"CONTEXT WINDOW: {config.default_llm_config.context_window}")
    print(f"MODEL WRAPPER: {config.default_llm_config.model_wrapper}")
    print(f"PRESET: {config.preset}")
    print(f"PROFILE: {config.persona}")
    print(f"USER: {config.human}")

    print(
        f"\n\t-> Final score: {total_points}/{len(constants.TEST_QUERIES) * constants.ITERATIONS}, total time: {runtime} seconds"
    )


def main():
    parser = argparse.ArgumentParser(description="Execute performance tests")
    parser.add_argument(
        "-m",
        "--messages",
        action="store_true",
        help="Display function calls and bot responses.",
    )
    parser.add_argument(
        "-n",
        "--n-tries",
        type=int,
        default=constants.ITERATIONS,
        help="Number of test iterations per function.",
    )
    args = parser.parse_args()

    constants.ITERATIONS = args.n_tries
    evaluate(args.messages)


if __name__ == "__main__":
    main()
