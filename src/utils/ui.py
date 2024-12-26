import re
from abc import ABC, abstractmethod
from typing import List, Optional

from colorama import Fore, Style, init

init(autoreset=True)

VERBOSE_OUTPUT = False
MINIMAL_DISPLAY = False


class MessageProcessor(ABC):
    @abstractmethod
    def handle_input(self, content: str, metadata: Optional["DataPacket"] = None):
        raise NotImplementedError

    @abstractmethod
    def process_thoughts(self, content: str, metadata: Optional["DataPacket"] = None):
        raise NotImplementedError

    @abstractmethod
    def generate_response(self, content: str, metadata: Optional["DataPacket"] = None):
        raise NotImplementedError

    @abstractmethod
    def execute_operation(self, content: str, metadata: Optional["DataPacket"] = None):
        raise NotImplementedError


class TerminalHandler(MessageProcessor):
    @staticmethod
    def display_critical(content: str):
        output_fmt = f"{Fore.MAGENTA}{Style.BRIGHT}{{text}}{Style.RESET_ALL}"
        if MINIMAL_DISPLAY:
            output_fmt = "{text}"
        print(output_fmt.format(text=content))

    @staticmethod
    def display_error(content: str):
        output_fmt = f"{Fore.RED}{Style.BRIGHT}{{text}}{Style.RESET_ALL}"
        if MINIMAL_DISPLAY:
            output_fmt = "{text}"
        print(output_fmt.format(text=content))

    @staticmethod
    def process_thoughts(content: str, metadata: Optional["DataPacket"] = None):
        output_fmt = f"\x1B[3m{Fore.LIGHTBLACK_EX}💭 {{text}}{Style.RESET_ALL}"
        if MINIMAL_DISPLAY:
            output_fmt = "{text}"
        print(output_fmt.format(text=content))

    @staticmethod
    def generate_response(content: str, metadata: Optional["DataPacket"] = None):
        output_fmt = (
            f"{Fore.YELLOW}{Style.BRIGHT}🤖 {Fore.YELLOW}{{text}}{Style.RESET_ALL}"
        )
        if MINIMAL_DISPLAY:
            output_fmt = "{text}"
        print(output_fmt.format(text=content))

    @staticmethod
    def display_memory(content: str, metadata: Optional["DataPacket"] = None):
        output_fmt = f"{Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}💾 {Fore.LIGHTMAGENTA_EX}{{text}}{Style.RESET_ALL}"
        if MINIMAL_DISPLAY:
            output_fmt = "{text}"
        print(output_fmt.format(text=content))

    @staticmethod
    def display_system(content: str, metadata: Optional["DataPacket"] = None):
        output_fmt = f"{Fore.MAGENTA}{Style.BRIGHT}⚙️ [kernel] {Fore.MAGENTA}{content}{Style.RESET_ALL}"
        if MINIMAL_DISPLAY:
            output_fmt = "{text}"
        print(output_fmt.format(text=content))

    @staticmethod
    def handle_input(
        content: str,
        metadata: Optional["DataPacket"] = None,
        raw_format: bool = False,
        dump_data: bool = False,
        verbose: bool = VERBOSE_OUTPUT,
    ):
        def render_input(symbol, text, output_fn=print):
            if MINIMAL_DISPLAY:
                output_fn(f"{symbol} {text}")
            else:
                output_fn(
                    f"{Fore.GREEN}{Style.BRIGHT}{symbol} {Fore.GREEN}{text}{Style.RESET_ALL}"
                )

        if not (raw_format or dump_data or verbose):
            return

        if isinstance(content, str):
            try:
                data = eval(content) if raw_format else json_loads(content)
            except:
                render_input("👤", content)
                return

        packet_type = data.get("type", "")
        if packet_type == "input_message":
            if dump_data:
                render_input("👤", data["content"])
            else:
                data.pop("type")
                render_input("👤", data)
        elif packet_type == "alive_signal":
            if verbose:
                data.pop("type")
                render_input("❤️", data)
        elif packet_type == "kernel_message":
            data.pop("type")
            render_input("⚙️", data)
        else:
            render_input("👤", data)

    @staticmethod
    def execute_operation(
        content: str,
        metadata: Optional["DataPacket"] = None,
        verbose: bool = VERBOSE_OUTPUT,
    ):
        def render_operation(symbol, text, color=Fore.RED, output_fn=print):
            if MINIMAL_DISPLAY:
                output_fn(f"⚡{symbol} [operation] {text}")
            else:
                output_fn(
                    f"{color}{Style.BRIGHT}⚡{symbol} [operation] {color}{text}{Style.RESET_ALL}"
                )

        if isinstance(content, dict):
            render_operation("", content) if verbose else None
            return

        if "Success" in content:
            render_operation("✅", content)
        elif "Error: " in content:
            render_operation("❌", content)
        elif content.startswith(("Running ", "Ran ")):
            operation_match = re.search(r"Running (\w+)\((.*)\)", content)
            if operation_match:
                op_name, op_args = operation_match.groups()
                if any(
                    x in op_name
                    for x in [
                        "memory_write",
                        "memory_read",
                        "memory_update",
                        "memory_append",
                    ]
                ):
                    try:
                        args_dict = eval(op_args)
                        self._process_memory_operation(op_name, args_dict)
                    except Exception as e:
                        print(f"Operation processing failed: {e}")
            elif verbose:
                render_operation("", content)
        else:
            try:
                data = json_loads(content)
                color = Fore.GREEN if data.get("status") == "OK" else Fore.RED
                render_operation("", str(data), color=color) if verbose else None
            except:
                render_operation("", content)

    @staticmethod
    def _process_memory_operation(op_name: str, args: dict):
        if "search" in op_name:
            output = f'\tquery: {args["query"]}, page: {args["page"]}'
        elif "write" in op_name:
            output = f'\t→ {args["content"]}'
        else:
            output = f'\t {args["old_content"]}\n\t→ {args["new_content"]}'

        if MINIMAL_DISPLAY:
            print(output)
        else:
            print(f"{Style.BRIGHT}{Fore.RED}{output}{Style.RESET_ALL}")

    @staticmethod
    def display_sequence(sequence: List["DataPacket"], dump_data: bool = False):
        for idx, packet in enumerate(reversed(sequence) if dump_data else sequence):
            if dump_data:
                print(f"[{len(sequence) - idx}] ", end="")

            data = packet.to_dict()
            role = data["role"]
            content = data["content"]

            handler_map = {
                "system": TerminalHandler.display_system,
                "assistant": lambda c: (
                    TerminalHandler.process_thoughts(c)
                    if data.get("tool_calls")
                    else TerminalHandler.generate_response(c)
                ),
                "user": lambda c: TerminalHandler.handle_input(c, dump=dump_data),
                "function": lambda c: TerminalHandler.execute_operation(
                    c, debug=dump_data
                ),
                "tool": lambda c: TerminalHandler.execute_operation(c, debug=dump_data),
            }

            handler = handler_map.get(role, lambda c: print(f"Unknown type: {c}"))
            handler(content)

    @staticmethod
    def display_basic_sequence(sequence: List["DataPacket"]):
        for packet in sequence:
            data = packet.to_dict()
            content = data["content"]

            if data["role"] == "system":
                TerminalHandler.display_system(content)
            elif data["role"] == "assistant":
                TerminalHandler.generate_response(content)
            elif data["role"] == "user":
                TerminalHandler.handle_input(content, raw_format=True)
            else:
                print(f"Unknown type: {content}")

    @staticmethod
    def display_raw_sequence(sequence: List["DataPacket"]):
        for packet in sequence:
            print(packet.to_dict())

    @staticmethod
    def signal_progress():
        pass

    @staticmethod
    def signal_completion():
        pass
