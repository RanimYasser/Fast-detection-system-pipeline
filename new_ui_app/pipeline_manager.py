import Model.newcomm2old as comm

class PipelineManager:
    def __init__(self):
        self.pipe = None

    def start(self, frame_q, info_q, batch_name=None):
        if self.pipe is not None:
            return
        self.pipe = comm.start_pipeline(
            label_cam_index=comm.LABEL_CAM_INDEX,
            frame_q=frame_q,
            info_q=info_q,
            batch_name=batch_name
        )

    def stop(self):
        if self.pipe is None:
            return
        comm.stop_pipeline(self.pipe)
        self.pipe = None
