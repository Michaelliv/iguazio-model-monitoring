import asyncio

import pandas as pd
from storey import WriteToParquet


class PathResolverParquetWriter(WriteToParquet):
    async def _emit(self, batch, batch_time):
        if self._first_event:
            await asyncio.get_running_loop().run_in_executor(None, self._makedirs)
            self._first_event = False
        df_columns = []
        df_columns.extend(self._columns)
        if self._index_cols:
            df_columns.extend(self._index_cols)
        df = pd.DataFrame(batch, columns=df_columns)
        if self._index_cols:
            df.set_index(self._index_cols, inplace=True)
        df.to_parquet(
            path=self._path,
            index=bool(self._index_cols),
            partition_cols=self._partition_cols,
        )
