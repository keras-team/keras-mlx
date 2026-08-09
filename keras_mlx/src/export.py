from keras.src.export.saved_model_export_archive import SavedModelExportArchive


class MlxExportArchive(SavedModelExportArchive):
    def track(self, resource):
        raise NotImplementedError(
            "`track` is not implemented in the mlx backend."
        )

    def add_endpoint(self, name, fn, input_signature=None, **kwargs):
        raise NotImplementedError(
            "`add_endpoint` is not implemented in the mlx backend."
        )
