# %pylab

from sklearn.decomposition import PCA

reload(induce)
ind = induce.Induce("dumps/short")
ind.read_embs("dumps/short_emb.npy", "dumps/short_labels")

data = ind.embeds
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(n_clusters=4, n_init=10).fit(data)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_)
