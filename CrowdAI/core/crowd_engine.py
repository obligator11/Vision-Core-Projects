import numpy as np
from sklearn.cluster import DBSCAN

class CrowdEngine:
    """
    Algorithmic spatial processing engine managing perspective-corrected ground plane 
    conversions, DBSCAN cluster aggregation, and rolling frame vector path metrics.
    """
    def __init__(self, eps_radius=75, min_group_samples=3, maximum_history_len=15):
        self.eps = eps_radius
        self.min_samples = min_group_samples
        self.max_history = maximum_history_len
        self.previous_centroids = None
        self.velocity_history = []

    def extract_ground_plane_coordinates(self, boxes):
        """
        Extract lower baseline centers (feet plane contacts) to prevent perspective 
        lens distortions from twisting proximity equations.
        """
        centroids = []
        for (x, y, w, h) in boxes:
            cx = int(x + (w / 2))
            cy = int(y + h)  # Base contact node
            centroids.append([cx, cy])
        return np.array(centroids)

    def process_frame_telemetry(self, boxes):
        """
        Processes multi-entity spatial patterns, grouping metrics, and flow rates.
        """
        total_occupants = len(boxes)
        
        if total_occupants == 0:
            return {
                "density_score": 0,
                "congestion_level": "LOW",
                "flow_efficiency": 100,
                "centroids": np.array([]),
                "cluster_labels": np.array([])
            }

        centroids = self.extract_ground_plane_coordinates(boxes)
        
        # ----------------------------------------------------------------
        # SPATIAL DENSITY CLUSTERING (DBSCAN)
        # ----------------------------------------------------------------
        # DBSCAN effectively tracks irregular, organic queue lines without
        # assuming a clean geometric center like K-Means does.
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(centroids)
        labels = db.labels_
        
        high_density_nodes = np.count_nonzero(labels != -1)
        
        # Density percentage calculation modeling
        density_score = int((high_density_nodes / total_occupants) * 100)
        
        # Algorithmic hazard zoning classification logic
        if total_occupants >= 12 or density_score >= 65:
            congestion_level = "HIGH CONGESTION"
        elif total_occupants >= 6 or density_score >= 30:
            congestion_level = "MEDIUM"
        else:
            congestion_level = "LOW"

        # ----------------------------------------------------------------
        # FRAME-TO-FRAME MOTION VELOCITY VECTORS
        # ----------------------------------------------------------------
        flow_efficiency = 100
        if self.previous_centroids is not None and len(self.previous_centroids) > 0:
            displacements = []
            for current_node in centroids:
                # Map vector path distances relative to matching historical coordinates
                deltas = np.linalg.norm(self.previous_centroids - current_node, axis=1)
                closest_match_idx = np.argmin(deltas)
                
                # Apply an upper tracking threshold to clean up coordinate switching noise
                if deltas[closest_match_idx] < 120:
                    displacements.append(deltas[closest_match_idx])
            
            if displacements:
                mean_frame_velocity = np.mean(displacements)
                self.velocity_history.append(mean_frame_velocity)
                if len(self.velocity_history) > self.max_history:
                    self.velocity_history.pop(0)
                
                smoothed_velocity = np.mean(self.velocity_history)
                
                # Flow drop detection matching static congestion patterns
                if smoothed_velocity < 2.0 and congestion_level != "LOW":
                    flow_efficiency = max(5, int(smoothed_velocity * 30))
                else:
                    # Dynamically compute movement efficiency scaling up to a 100% cap
                    flow_efficiency = min(100, int(55 + (smoothed_velocity * 3.5)))
            else:
                if congestion_level == "HIGH CONGESTION":
                    flow_efficiency = 10  # Frozen human structural blockade
        else:
            if congestion_level == "HIGH CONGESTION":
                flow_efficiency = 20
                
        self.previous_centroids = centroids.copy()

        return {
            "density_score": density_score,
            "congestion_level": congestion_level,
            "flow_efficiency": flow_efficiency,
            "centroids": centroids,
            "cluster_labels": labels
        }