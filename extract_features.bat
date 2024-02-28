ECHO dense train 5
C:\Users\hidri\miniconda3\python.exe save_feat.py name=dense_5 config=C:\Users\hidri\AML\AML-egovision\configs\I3D_save_feat.yaml dataset.shift=D1-D1 dataset.RGB.data_path=C:\Users\hidri\AML\ek_data\frames split=train dataset.workers=0 save.num_frames_per_clip.RGB=5 save.dense_sampling.RGB=True
ECHO dense test 5
C:\Users\hidri\miniconda3\python.exe save_feat.py name=dense_5 config=C:\Users\hidri\AML\AML-egovision\configs\I3D_save_feat.yaml dataset.shift=D1-D1 dataset.RGB.data_path=C:\Users\hidri\AML\ek_data\frames split=test dataset.workers=0 save.num_frames_per_clip.RGB=5 save.dense_sampling.RGB=True
PAUSE