import numpy as np
from sklearn.cluster import KMeans

def cluster_customers(customer_df, depot_df, num_clusters, plot=False):
    # 1. Extract coordinates
    cust_coords = customer_df[['latitude', 'longitude']].values

    # 2. Run KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    customer_df['cluster_id'] = kmeans.fit_predict(cust_coords)

    # 3. Assign nearest depot to each cluster
    cluster_to_depot = {}
    for cid in range(num_clusters):
        centroid = kmeans.cluster_centers_[cid]

        depot_df['dist'] = np.sqrt(
            (depot_df['latitude'] - centroid[0])**2 +
            (depot_df['longitude'] - centroid[1])**2
        )

        nearest_depot = depot_df.loc[depot_df['dist'].idxmin()]
        cluster_to_depot[cid] = nearest_depot['id_depot']

    # 4. Map depot assignment back to customers
    customer_df['assigned_depot'] = customer_df['cluster_id'].map(cluster_to_depot)

    # 5. Optional plotting (kept commented)
    # if plot:
    #     from Visuals.visualise_clusters import plot_clusters
    #     plot_clusters(customer_df, depot_df, kmeans)

    return customer_df, kmeans, cluster_to_depot