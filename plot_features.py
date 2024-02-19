import os
import pickle as pk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from omegaconf import OmegaConf
from collections import defaultdict

def plot_features(args):
    features_path = args.get('features_path', os.path.join('./saved_features/SAVE_DENSE-extracted_D1_test.pkl'))
    samples_path = args.get('split_path', os.path.join('./train_val/D1_test.pkl'))
    base_image_path = args.get('images_path', os.path.join('../ek_data/frames/'))
    output_image_path = args.get('output_image_path', os.path.join('plots/test.png'))
    
    plot_3d = args.get('plot_3D', False)
    
    reduced_features = []
    central_frames = []
    actions = []

    with open(features_path, 'rb') as f_file:
        data = pk.load(f_file)
        pca = PCA(3 if plot_3d else 2)
        
        with open(samples_path, 'rb') as s_file:
            samples = pk.load(s_file)
        
        # dictionary of label: list_of_action
        label_actions = defaultdict(set)
        for idx in range(len(samples)):
            label_actions[samples['verb_class'][idx]].add(samples['verb'][idx])

        for label, acts in label_actions.items():
            label_actions[label] = ', '.join(acts)

        for d in data['features']:
            features = d['features_RGB']
            features: np.ndarray = np.mean(features, 0)
            reduced_features.append(features)

            s = [t for t, l in enumerate(samples['uid']) if l == d['uid']][0]
            sample_central_frame = samples['start_frame'][s] + (samples['stop_frame'][s] - samples['start_frame'][s])//2
            
            label = samples['verb_class'][s]
            actions.append(label_actions[label])

            # TODO Generalize images naming
            img_path = os.path.join(base_image_path, f"{d['video_name']}/img_{sample_central_frame:010d}.jpg")
            central_frames.append(img_path)

    reduced_features = pca.fit_transform(reduced_features)
    reduced_features = np.array(reduced_features)
    reduced_features = reduced_features.reshape(reduced_features.shape[0], 3 if plot_3d else 2)
    
    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots()
    
    # TODO Generalize number of clusters
    km = KMeans(n_clusters=8, random_state=62)
    km.fit(reduced_features)
    
    predictions = km.predict(reduced_features)
    
    if not plot_3d and args.get('use_frames', False):
        for i, coord in enumerate(reduced_features):
            imagebox = OffsetImage(plt.imread(central_frames[i]), zoom=0.06)
            ab = AnnotationBbox(imagebox, coord, frameon=False)
            ax.add_artist(ab)

    # Making a list of [(coordinates), action]
    coord_action = []
    for i, coord in enumerate(reduced_features):
        coord_action.append((actions[i], coord))

    # Group by the first element of each tuple
    grouped_data = defaultdict(list)
    for action, coordinates in coord_action:
        grouped_data[action].append(coordinates)

    average_coord_per_action = defaultdict()
    for action, coordinates in grouped_data.items():
        # Extract x-coordinates and y-coordinates into separate lists
        x_coordinates = [x for x, y in coordinates]
        y_coordinates = [y for x, y in coordinates]
        # Calculate the mean of x-coordinates and y-coordinates
        x_mean = sum(x_coordinates) / len(coordinates)
        y_mean = sum(y_coordinates) / len(coordinates)
        average_coord_per_action[action] = (x_mean, y_mean)

    if plot_3d:
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=predictions)
    else:
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predictions)

    if not plot_3d:
        # Define a list of markers
        markers = ['o', '^', 's', 'p', 'P', '*', 'D', 'v']
        marker_iter = iter(markers)

        # Plot each point with a different marker
        for action, coordinate in average_coord_per_action.items():
            x, y = coordinate
            marker = next(marker_iter)
            plt.scatter(x, y, marker=marker, s=200, edgecolor='black', linewidths=2, label=action)
            plt.legend(fontsize='small', markerscale=0.7)

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(output_image_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    cli_args = OmegaConf.from_cli()
    print(cli_args)
    plot_features(cli_args)