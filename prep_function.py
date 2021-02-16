from os import environ

from mlrun import code_to_function, mount_v3io
from mlrun.config import config
from mlrun.runtimes import RemoteRuntime


class ModelmonitoringFunction:
    @staticmethod
    def deploy():
        fn: RemoteRuntime = code_to_function(
            name="model-monitoring",
            project="projects",
            filename="model_monitoring.py",
            kind="nuclio",
        )

        fn.spec.build.commands = [
            "pip install git+https://github.com/Michaelliv/iguazio-model-monitoring.git",
            "pip install v3io_frames",
            "pip install v3io-py",
        ]

        fn.set_envs(
            {
                "V3IO_ACCESS_KEY": environ.get("V3IO_ACCESS_KEY"),
                "V3IO_FRAMESD": environ.get("V3IO_FRAMESD"),
                "V3IO_API": environ.get("V3IO_API"),
            }
        )

        stream_path = config.model_endpoint_monitoring.stream_url.format(
            project="projects"
        )

        fn.add_v3io_stream_trigger(
            stream_path=stream_path, name="monitoring_stream_trigger",
        )

        fn.apply(mount_v3io())
        fn.deploy()
