class DummyExp:
    def __init__(self):
        self.args = DummyArgs({
			"batch_size": 32,
			"num_workers": 5,
            "train_patients": train_patients,
            "val_patients": val_patients,
            "test_patients": test_patients,
		})

    def log_console(self, *args, **kwargs):
	    print(self, *args, **kwargs)