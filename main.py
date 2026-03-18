import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
 
GESTURES = ['wave', 'thumbs_up', 'peace', 'point', 'fist', 'open_hand']
KEYPOINTS = 21  # Hand keypoints (MediaPipe)
 
def extract_angle_features(keypoints):
    """Extract joint angles from keypoint coordinates."""
    kp = np.array(keypoints).reshape(-1, 3)
    features = []
    # Finger angles
    finger_indices = [(0,1,2),(1,2,3),(2,3,4),   # thumb
                       (0,5,6),(5,6,7),(6,7,8),    # index
                       (0,9,10),(9,10,11),(10,11,12), # middle
                       (0,13,14),(13,14,15),(14,15,16), # ring
                       (0,17,18),(17,18,19),(18,19,20)] # pinky
    for a, b, c in finger_indices:
        if max(a,b,c) < len(kp):
            v1 = kp[a] - kp[b]; v2 = kp[c] - kp[b]
            cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
            features.append(np.clip(cos,-1,1))
    # Relative distances from wrist
    wrist = kp[0]
    for i in [4,8,12,16,20]:
        if i < len(kp):
            features.append(np.linalg.norm(kp[i]-wrist))
    return features
 
def simulate_gesture_data(n=300):
    X, y = [], []
    np.random.seed(42)
    for label, g in enumerate(GESTURES):
        for _ in range(n//len(GESTURES)):
            kp = np.random.randn(KEYPOINTS, 3)
            kp[0] = [0,0,0]  # wrist
            feats = extract_angle_features(kp.flatten())
            X.append(feats); y.append(label)
    return np.array(X), np.array(y)
 
X, y = simulate_gesture_data()
scaler = StandardScaler(); X_s = scaler.fit_transform(X)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_s, y, cv=5)
clf.fit(X_s, y)
print(f"Gesture Recognition Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
test = np.random.randn(KEYPOINTS*3); test_f = extract_angle_features(test)
pred = clf.predict(scaler.transform([test_f]))[0]
print(f"Sample prediction: {GESTURES[pred]}")
