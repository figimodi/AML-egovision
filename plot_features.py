import os
import pickle as pk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from omegaconf import OmegaConf

def plot_features(args):
    features_path = args.get('features_path', os.path.join('./saved_features/SAVE_DENSE-extracted_D1_test.pkl'))
    samples_path = args.get('split_path', os.path.join('./train_val/D1_test.pkl'))
    base_image_path = args.get('images_path', os.path.join('../ek_data/frames/'))
    output_image_path = args.get('output_image_path', os.path.join('plots/test.png'))
    
    plot_3d = args.get('plot_3D', False)
    
    central_features = []
    central_frames = []
    
    with open(features_path, 'rb') as f_file:
        data = pk.load(f_file)
        pca = PCA(3 if plot_3d else 2)
        
        with open(samples_path, 'rb') as s_file:
            samples = pk.load(s_file)
        
        for d in data['features']:
            features = d['features_RGB']
            reduced_features = pca.fit_transform(features)
            central_rf: np.ndarray = reduced_features[len(reduced_features)//2].reshape(1, -1)
            
            central_features.append(central_rf)
            
            s = [t for t, l in enumerate(samples['uid']) if l == d['uid']][0]
            sample_central_frame = samples['stop_frame'][s] - samples['start_frame'][s]
            
            # TODO Generalize images naming
            img_path = os.path.join(base_image_path, f"{d['video_name']}/img_{sample_central_frame:010d}.jpg")
            central_frames.append(img_path)
    
    central_features = np.array(central_features)
    central_features = central_features.reshape(central_features.shape[0], 3 if plot_3d else 2)
    
    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots()
    
    # TODO Generalize number of clusters
    km = KMeans(n_clusters=8, random_state=62)
    km.fit(central_features)
    
    predictions = km.predict(central_features)
    
    if not plot_3d and args.get('use_frames', False):
        for i, coord in enumerate(central_features):
            imagebox = OffsetImage(plt.imread(central_frames[i]), zoom=0.06)
            ab = AnnotationBbox(imagebox, coord, frameon=False)
            ax.add_artist(ab)
            
    if plot_3d:
        ax.scatter(central_features[:, 0], central_features[:, 1], central_features[:, 2], c=predictions)
    else:
        ax.scatter(central_features[:, 0], central_features[:, 1], c=predictions)
        
    plt.savefig(output_image_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    cli_args = OmegaConf.from_cli()
    print(cli_args)
    plot_features(cli_args)