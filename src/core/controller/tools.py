import ast
import base64
import io
import os
import pickle
import runpy
import subprocess
import sys
import tempfile
import traceback
import uuid
import venv
from typing import Any, Dict, Optional

log = create_logger(__name__)


class ExecutionEnvironment:
    STATE_KEY = "env_state"
    REQ_FILE_NAME = "dependencies.txt"

    HASH_NAMESPACE = uuid.NAMESPACE_DNS
    RESULT_START_TAG = str(uuid.uuid5(HASH_NAMESPACE, "result-start"))
    RESULT_END_TAG = str(uuid.uuid5(HASH_NAMESPACE, "result-end"))

    EXECUTION_VAR_NAME = "executionResult_5gYjdzb3"

    def __init__(
        self,
        tool_name: str,
        parameters: dict,
        user: User,
        recreate=False,
        tool_instance: Optional[Tool] = None,
    ):
        self.tool_name = tool_name
        self.parameters = parameters
        self.user = user

        if tool_instance:
            self.tool = tool_instance
        else:
            self.tool = ToolManager().fetch_tool_by_name(
                tool_name=self.tool_name, actor=self.user
            )
            if not self.tool:
                raise ValueError(
                    f"Attempted to execute a non-existent tool '{self.tool_name}' for user {self.user.organization_id}"
                )

        self.config_manager = ConfigManager(tool_settings)
        self.recreate_env = recreate

    def execute(self, agent_status: Optional[AgentState] = None) -> ExecutionResult:
        if tool_settings.api_key:
            log.debug(f"Using remote sandbox for executing {self.tool_name}")
            result = self.execute_remote(agent_status=agent_status)
        else:
            log.debug(f"Using local sandbox for executing {self.tool_name}")
            result = self.execute_local(agent_status=agent_status)

        log.debug(f"Executed '{self.tool_name}', capturing execution output: \n")
        for output in (result.stdout or []) + (result.stderr or []):
            log.debug(f"{output}")
        log.debug("Execution output logged.")

        return result

    @contextmanager
    def modify_env(self, new_vars: dict):
        original_env = os.environ.copy()
        os.environ.update(new_vars)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def execute_local(self, agent_status: AgentState) -> ExecutionResult:
        environment_config = self.config_manager.get_or_create_config_for_type(
            EnvironmentType.LOCAL, self.user
        )
        local_settings = environment_config.get_local_settings()

        env_vars = self.config_manager.get_environment_variables(
            environment_id=environment_config.id, actor=self.user, limit=100
        )
        env = os.environ.copy()
        env.update(env_vars)

        if not os.path.isdir(local_settings.sandbox_dir):
            raise FileNotFoundError(
                f"Sandbox directory is missing: {local_settings.sandbox_dir}"
            )

        with tempfile.NamedTemporaryFile(
            mode="w", dir=local_settings.sandbox_dir, suffix=".py", delete=False
        ) as temp_script:
            code = (
                self.create_execution_script(
                    agent_status=agent_status, wrap_output=True
                )
                if local_settings.use_virtualenv
                else self.create_execution_script(agent_status=agent_status)
            )
            temp_script.write(code)
            temp_script.flush()
            script_path = temp_script.name

        try:
            return (
                self.run_local_with_venv(environment_config, env, script_path)
                if local_settings.use_virtualenv
                else self.run_local_with_runpy(
                    environment_config, env_vars, script_path
                )
            )
        except Exception as err:
            log.error(f"Error during execution of '{self.tool_name}': {err}")
            log.error(f"Code for debugging:\n{code}")
            raise err
        finally:
            os.remove(script_path)

    def run_local_with_venv(
        self,
        environment_config: EnvironmentConfig,
        environment: Dict[str, str],
        script_path: str,
    ) -> ExecutionResult:
        local_settings = environment_config.get_local_settings()
        venv_path = os.path.join(
            local_settings.sandbox_dir, local_settings.virtualenv_name
        )

        if not os.path.isdir(venv_path):
            log.warning(
                f"Virtual environment not found at {venv_path}, creating one..."
            )
            self.setup_virtualenv_for_local(
                environment_config.sandbox_dir, venv_path, environment
            )

        python_exec = os.path.join(venv_path, "bin", "python3")
        if not os.path.isfile(python_exec):
            raise FileNotFoundError(
                f"Python executable not found in virtual environment at {python_exec}"
            )

        environment["VIRTUAL_ENV"] = venv_path
        environment["PATH"] = f"{os.path.join(venv_path, 'bin')}:{environment['PATH']}"
        environment["PYTHONWARNINGS"] = "ignore"

        try:
            process_result = subprocess.run(
                [python_exec, script_path],
                env=environment,
                cwd=local_settings.sandbox_dir,
                timeout=60,
                capture_output=True,
                text=True,
            )
            execution_output, stdout = self.extract_result_from_output(
                process_result.stdout
            )
            function_output, agent_status = self.extract_best_result(execution_output)
            return ExecutionResult(
                function_output=function_output,
                agent_status=agent_status,
                stdout=[stdout] if stdout else [],
                stderr=[process_result.stderr] if process_result.stderr else [],
                status="success",
                config_fingerprint=environment_config.fingerprint(),
            )

        except subprocess.CalledProcessError as error:
            log.error(f"Error executing tool '{self.tool_name}': {error}")
            friendly_error_msg = create_friendly_error_message(
                tool_name=self.tool_name,
                error_name=type(error).__name__,
                error_message=str(error),
            )
            return ExecutionResult(
                function_output=friendly_error_msg,
                agent_status=None,
                stdout=[error.stdout] if error.stdout else [],
                stderr=[error.stderr] if error.stderr else [],
                status="error",
                config_fingerprint=environment_config.fingerprint(),
            )

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Tool execution '{self.tool_name}' timed out.")

        except Exception as err:
            log.error(f"Unexpected error during execution of '{self.tool_name}': {err}")
            raise err

    def run_local_with_runpy(
        self,
        environment_config: EnvironmentConfig,
        env_vars: Dict[str, str],
        script_path: str,
    ) -> ExecutionResult:
        status = "success"
        agent_status, stderr = None, None

        old_stdout, old_stderr = sys.stdout, sys.stderr
        captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = captured_stdout, captured_stderr

        try:
            with self.modify_env(env_vars):
                result = runpy.run_path(script_path, init_globals=env_vars)

            function_result = result.get(self.EXECUTION_VAR_NAME)
            function_output, agent_status = self.extract_best_result(function_result)

        except Exception as err:
            function_output = create_friendly_error_message(
                function_name=self.tool_name,
                error_name=type(err).__name__,
                error_message=str(err),
            )
            traceback.print_exc(file=sys.stderr)
            status = "error"

        sys.stdout, sys.stderr = old_stdout, old_stderr
        stdout_output = (
            [captured_stdout.getvalue()] if captured_stdout.getvalue() else []
        )
        stderr_output = (
            [captured_stderr.getvalue()] if captured_stderr.getvalue() else []
        )

        return ExecutionResult(
            function_output=function_output,
            agent_status=agent_status,
            stdout=stdout_output,
            stderr=stderr_output,
            status=status,
            config_fingerprint=environment_config.fingerprint(),
        )

    def extract_result_from_output(self, text: str):
        if self.RESULT_START_TAG not in text:
            return "", text
        start_idx = text.index(self.RESULT_START_TAG) + len(self.RESULT_START_TAG)
        end_idx = text.index(self.RESULT_END_TAG)
        return (
            text[start_idx:end_idx],
            text[: start_idx - len(self.RESULT_START_TAG)]
            + text[end_idx + len(self.RESULT_END_TAG) :],
        )

    def setup_virtualenv_for_local(
        self, sandbox_dir: str, venv_path: str, env: Dict[str, str]
    ):
        venv.create(venv_path, with_pip=True)

        pip_path = os.path.join(venv_path, "bin", "pip")
        try:
            log.info("Upgrading pip in the virtual environment...")
            subprocess.run(
                [pip_path, "install", "--upgrade", "pip"], env=env, check=True
            )

            req_file_path = os.path.join(sandbox_dir, self.REQ_FILE_NAME)
            if os.path.isfile(req_file_path):
                log.info(f"Installing dependencies from {req_file_path}")
                subprocess.run(
                    [pip_path, "install", "-r", req_file_path], env=env, check=True
                )
                log.info("Dependencies installed successfully.")
            else:
                log.warning(
                    "No 'dependencies.txt' file found. Skipping dependency installation."
                )

        except subprocess.CalledProcessError as err:
            log.error(f"Error setting up virtual environment: {err}")
            raise RuntimeError(f"Failed to configure virtual environment: {err}")

    def execute_remote(self, agent_status: AgentState) -> ExecutionResult:
        environment_config = self.config_manager.get_or_create_config_for_type(
            EnvironmentType.REMOTE, self.user
        )
        remote_instance = self.create_or_fetch_remote_instance(environment_config)
        if not remote_instance or self.recreate_env:
            remote_instance = self.initialize_remote_instance(environment_config)

        remote_instance.set_timeout(environment_config.get_remote_config().timeout)

        env_vars = self.config_manager.get_environment_variables(
            environment_id=environment_config.id, actor=self.user, limit=100
        )
        code = self.create_execution_script(agent_status=agent_status)
        remote_result = remote_instance.run_script(
            code=code,
            environment_variables=env_vars,
            exec_id=self.tool_name,
        )

        return remote_result

    def create_or_fetch_remote_instance(self, config: EnvironmentConfig):
        # Return a pre-existing instance if available, otherwise create a new one
        pass

    def initialize_remote_instance(self, config: EnvironmentConfig):
        # Setup a new remote environment for execution
        pass

    def create_execution_script(
        self, agent_status: Optional[AgentState], wrap_output=False
    ) -> str:
        # Return the full script that will be executed either locally or remotely
        pass
