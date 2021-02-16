from mlrun.run import MLClientCtx
from nuclio import Event

from monitoring.stream import EventStreamProcessor


def init_context(context: MLClientCtx):
    context.logger.info("Initializing EventStreamProcessor")
    stream_processor = EventStreamProcessor()
    setattr(context, "stream_processor", stream_processor)


def handler(context: MLClientCtx, event: Event):
    print(hasattr(context, "stream_processor"))
    context.logger.info(str(event))
