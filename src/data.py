import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

class FER2013(Dataset):
    def __init__(self, df, augment=False):
        self.pixels = df['pixels'].values
        self.labels = df['emotion'].values.astype(np.int64)
        aug = [T.RandomHorizontalFlip(), T.RandomRotation(10),
               T.RandomAffine(0, translate=(0.1,0.1))] if augment else []
        self.tf = T.Compose(aug + [T.Normalize([0.5],[0.5])])
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        img = np.array(self.pixels[i].split(), dtype=np.float32).reshape(48,48)/255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return self.tf(img), self.labels[i]

def get_loaders(df, batch_size=64, augment=True, num_workers=0):
    tr = FER2013(df[df.Usage=='Training'],   augment=augment)
    va = FER2013(df[df.Usage=='PublicTest'], augment=False)
    te = FER2013(df[df.Usage=='PrivateTest'],augment=False)
    mk = lambda d,s: DataLoader(d, batch_size=batch_size, shuffle=s, num_workers=num_workers, pin_memory=True)
    return mk(tr,True), mk(va,False), mk(te,False)

def class_weights(df):
    counts = df[df.Usage=='Training']['emotion'].value_counts().sort_index().values
    return torch.tensor(counts.sum()/(len(counts)*counts), dtype=torch.float32)
