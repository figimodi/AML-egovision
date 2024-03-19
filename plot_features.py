import os
import pickle as pk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from omegaconf import OmegaConf
from collections import defaultdict

def plot_features_PCA(args):
    features_path = args.get('features_path', os.path.join('./saved_features/SAVE_DENSE-extracted_D1_test.pkl'))
    samples_path = args.get('split_path', os.path.join('./train_val/D1_test.pkl'))
    base_image_path = args.get('images_path', os.path.join('../ek_data/frames/'))
    output_image_path = args.get('output_image_path', os.path.join('plots/test.png'))
    
    plot_3d = args.get('plot_3D', False)
    
    reduced_features = []
    central_frames = []
    actions = []
    pca = PCA(3 if plot_3d else 2)

    with open(features_path, 'rb') as f_file:
        data = pk.load(f_file)
        
        with open(samples_path, 'rb') as s_file:
            samples = pk.load(s_file)
        
            # dictionary of label: list_of_action
            label_actions = defaultdict(set)
            for idx in range(len(samples)):
                label_actions[samples['verb_class'][idx]].add(samples['verb'][idx])

            for label, acts in label_actions.items():
                label_actions[label] = ', '.join(acts)

            reduced_features = [x['features_RGB'] for x in data['features']]
            reduced_features = np.mean(reduced_features, 1)

            labels = samples['verb_class']
            actions = [label_actions[label] for label in labels] 

            sample_central_frames = samples['start_frame'] + (samples['stop_frame'] - samples['start_frame'])//2

            video_names = [x['video_name'] for x in data['features']]
            central_frames = [os.path.join(base_image_path, f"{video_names[idx]}/img_{sample_central_frames[idx]:010d}.jpg") for idx in range(len(video_names))]

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

def plot_features_LDA(args):
    features_path = args.get('features_path', os.path.join('./saved_features/SAVE_DENSE-extracted_D1_test.pkl'))
    samples_path = args.get('split_path', os.path.join('./train_val/D1_test.pkl'))
    base_image_path = args.get('images_path', os.path.join('../ek_data/frames/'))
    output_image_path = args.get('output_image_path', os.path.join('plots/test.png'))

    NUM_CLASSES = 8
    extracted_samples = []
    labels = []
    lda = LinearDiscriminantAnalysis(n_components=2)
    
    with open(features_path, 'rb') as f_file:
        data = pk.load(f_file)
        
        with open(samples_path, 'rb') as s_file:
            samples = pk.load(s_file)
        
            # dictionary of label: list_of_action
            label_actions = defaultdict(set)
            for idx in range(len(samples)):
                label_actions[samples['verb_class'][idx]].add(samples['verb'][idx])

            for label, acts in label_actions.items():
                label_actions[label] = ', '.join(acts)

            extracted_samples = np.array([x['features_RGB'] for x in data['features']])
            extracted_samples = np.mean(extracted_samples, 1)
            labels = samples['verb_class']

    extracted_samples = lda.fit_transform(extracted_samples, labels)
    extracted_samples = np.array(extracted_samples)
    extracted_samples = extracted_samples.reshape(extracted_samples.shape[0],  2)
    
    color = iter(plt.cm.rainbow(np.linspace(0, 1, NUM_CLASSES)))

    for label in range(NUM_CLASSES):
        cl = next(color)
        class_samples = np.array([extracted_samples[j] for j in range(len(extracted_samples)) if labels[j] == label])
        plt.scatter(class_samples[:, 0], class_samples[:, 1], c=cl, label=label_actions[label])

    plt.legend(fontsize='small', loc='upper right')
    plt.gcf().set_size_inches(10, 7)
    plt.savefig(output_image_path, dpi=300)
    plt.show()

def plot_features_PCA_LDA(args):
    features_path = args.get('features_path', os.path.join('./saved_features/uniform_25_D1_test.pkl'))
    samples_path = args.get('split_path', os.path.join('./train_val/D1_test.pkl'))
    base_image_path = args.get('images_path', os.path.join('../ek_data/frames/'))
    output_image_path = args.get('output_image_path', os.path.join('plots/PCA_LDA/uniform/uniform_25_test_woim_PCA_LDA.png'))

    retained_variance = .98
    
    reduced_features_pca = []
        
    pca = PCA(n_components=200)
    data, samples = None, None

    with open(features_path, 'rb') as f_file, open(samples_path, 'rb') as s_file:
        data = pk.load(f_file)
        samples = pk.load(s_file)
    
    # dictionary of label: list_of_action
    label_actions = defaultdict(set)
    for idx in range(len(samples)):
        label_actions[samples['verb_class'][idx]].add(samples['verb'][idx])

    for label, acts in label_actions.items():
        label_actions[label] = ', '.join(acts)

    features = [x['features_RGB'] for x in data['features']]
    features = np.mean(features, 1)

    labels = samples['verb_class']

    reduced_features_pca = pca.fit_transform(features)

    for k, v in enumerate(pca.explained_variance_ratio_.cumsum()):
        if v >= retained_variance:
            break

    k += 1

    indexes = pca.explained_variance_.argsort()[::-1][:k]
    reduced_features_pca = reduced_features_pca[:,indexes]

    NUM_CLASSES = 8

    lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')

    print(reduced_features_pca.shape)
    print(len(labels))

    extracted_samples = lda.fit_transform(reduced_features_pca, labels, )
    
    color = iter(plt.cm.rainbow(np.linspace(0, 1, NUM_CLASSES)))

    for label in range(NUM_CLASSES):
        cl = next(color)
        class_samples = np.array([extracted_samples[j] for j in range(len(extracted_samples)) if labels[j] == label])
        plt.scatter(class_samples[:, 0], class_samples[:, 1], c=cl, label=label_actions[label])

    plt.legend(fontsize='small', loc='upper right')
    plt.gcf().set_size_inches(10, 7)
    plt.savefig(output_image_path, dpi=300)
    plt.show()

def plot_features_TSNE(args):
    features_path = args.get('features_path', os.path.join('./saved_features/SAVE_DENSE-extracted_D1_test.pkl'))
    samples_path = args.get('split_path', os.path.join('./train_val/D1_test.pkl'))
    base_image_path = args.get('images_path', os.path.join('../ek_data/frames/'))
    output_image_path = args.get('output_image_path', os.path.join('plots/test.png'))

    NUM_CLASSES = 8
    extracted_samples = []
    labels = []
    actions = []

    # use PCA or LDA before using t-sne
    PCA_COMPONENTS = 50
    pca = PCA(PCA_COMPONENTS)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30)

    with open(features_path, 'rb') as f_file:
        data = pk.load(f_file)
        
        with open(samples_path, 'rb') as s_file:
            samples = pk.load(s_file)
        
            # dictionary of label: list_of_action
            label_actions = defaultdict(set)
            for idx in range(len(samples)):
                label_actions[samples['verb_class'][idx]].add(samples['verb'][idx])

            for label, acts in label_actions.items():
                label_actions[label] = ', '.join(acts)

            extracted_samples = np.array([x['features_RGB'] for x in data['features']])
            extracted_samples = np.mean(extracted_samples, 1)
            
            labels = samples['verb_class']
            actions = [label_actions[label] for label in labels] 

    extracted_samples = pca.fit_transform(extracted_samples)
    extracted_samples = np.array(extracted_samples)
    extracted_samples = extracted_samples.reshape(extracted_samples.shape[0], PCA_COMPONENTS)

    # apply t-sne
    extracted_samples = tsne.fit_transform(extracted_samples)
    extracted_samples = np.array(extracted_samples)
    extracted_samples = extracted_samples.reshape(extracted_samples.shape[0], 2)

    # km = KMeans(n_clusters=8, random_state=62)
    # km.fit(extracted_samples)
    # predictions = km.predict(extracted_samples)
    # plt.scatter(extracted_samples[:, 0], extracted_samples[:, 1], c=predictions)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, NUM_CLASSES)))

    for label in range(NUM_CLASSES):
        cl = next(color)
        class_samples = np.array([extracted_samples[j] for j in range(len(extracted_samples)) if labels[j] == label])
        plt.scatter(class_samples[:, 0], class_samples[:, 1], c=cl, label=label_actions[label])

    # # Making a list of [(coordinates), action]
    # coord_action = []
    # for i, coord in enumerate(extracted_samples):
    #     coord_action.append((actions[i], coord))

    # # Group by the first element of each tuple
    # grouped_data = defaultdict(list)
    # for action, coordinates in coord_action:
    #     grouped_data[action].append(coordinates)

    # average_coord_per_action = defaultdict()
    # for action, coordinates in grouped_data.items():
    #     # Extract x-coordinates and y-coordinates into separate lists
    #     x_coordinates = [x for x, y in coordinates]
    #     y_coordinates = [y for x, y in coordinates]
    #     # Calculate the mean of x-coordinates and y-coordinates
    #     x_mean = sum(x_coordinates) / len(coordinates)
    #     y_mean = sum(y_coordinates) / len(coordinates)
    #     average_coord_per_action[action] = (x_mean, y_mean)

    # # Define a list of markers
    # markers = ['o', '^', 's', 'p', 'P', '*', 'D', 'v']
    # marker_iter = iter(markers)

    # # Plot each point with a different marker
    # for action, coordinate in average_coord_per_action.items():
    #     x, y = coordinate
    #     marker = next(marker_iter)
    #     plt.scatter(x, y, marker=marker, s=200, edgecolor='black', linewidths=2, label=action)
    #     plt.legend(fontsize='small', markerscale=0.7, loc="upper right")

    plt.legend(fontsize='small', markerscale=0.7, loc="lower right")
    plt.gcf().set_size_inches(12, 8)
    plt.savefig(output_image_path, dpi=300)
    plt.show()

    return

if __name__ == '__main__':
    cli_args = OmegaConf.from_cli()
    print(cli_args)
    # plot_features_PCA(cli_args)
    # plot_features_LDA(cli_args)
    # plot_features_PCA_LDA(cli_args)
    plot_features_TSNE(cli_args)

    # TODO: try isomap, density-based clustering vs actual labels, use some different clustering algorithms from k-means (DBSCAN)