import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import re
import random
from torch.optim import Adam

# set random seed biar hasil nya konsisten tiap running
random.seed(42)
torch.manual_seed(42)

def buat_data_email():
    """
    bikin data email palsu buat training. 50 email phishing 50 email normal
    """
    template_phishing = [
        "selamat anda menang undian {brand}! klik {url}",
        "akun {brand} anda terblokir, verifikasi di {url}",
        "pembayaran {brand} gagal, bayar segera di {url}",
        "pajak anda belum dibayar, bayar di {url}",
        "bonus pulsa {brand} menunggu, ambil di {url}"
    ]
    
    template_normal = [
        "jadwal kuliah {jurusan} minggu ini",
        "tagihan listrik bulan {bulan} sudah dibayar", 
        "reminder meeting project {nama} besok",
        "update status pengiriman {barang}",
        "konfirmasi pendaftaran {acara}"
    ]
    
    # data dummy buat variasi template
    brand = ["tokopedia", "shopee", "gojek", "dana", "ovo"]
    url = ["bit.ly/win22", "klik.me/vrf21", "pay.me/now", "web.co/pay"]
    jurusan = ["teknik elektro", "informatika", "mesin", "sipil"]
    bulan = ["januari", "februari", "maret", "april"]
    nama = ["website", "mobile app", "sistem erp", "database"]
    barang = ["laptop", "hp", "monitor", "printer"]
    acara = ["workshop ai", "seminar tech", "bootcamp coding", "hackathon"]
    
    emails = []
    labels = []
    
    # bikin 50 email phishing
    for _ in range(50):
        template = random.choice(template_phishing)
        email = template.format(
            brand=random.choice(brand),
            url=random.choice(url)
        )
        emails.append(email)
        labels.append(1)
    
    # bikin 50 email normal  
    for _ in range(50):
        template = random.choice(template_normal)
        email = template.format(
            jurusan=random.choice(jurusan),
            bulan=random.choice(bulan),
            nama=random.choice(nama),
            barang=random.choice(barang),
            acara=random.choice(acara)
        )
        emails.append(email)
        labels.append(0)
        
    return emails, labels

def proses_teks(emails):
    """
    proses email jadi format yang bisa dipakai model
    """
    # hapus tanda baca dan ubah ke lowercase
    emails_clean = []
    for email in emails:
        email = email.lower()
        email = re.sub(r'[^\w\s]', '', email)
        emails_clean.append(email)
    
    # bikin kamus kata
    semua_kata = []
    for email in emails_clean:
        semua_kata.extend(email.split())
    kata_unik = list(set(semua_kata))[:50]
    kamus = {kata: idx+1 for idx, kata in enumerate(kata_unik)}
    
    # ubah teks jadi angka pake kamus
    hasil = np.zeros((len(emails), 10))
    for i, email in enumerate(emails_clean):
        token = email.split()[:10]
        for j, kata in enumerate(token):
            if kata in kamus:
                hasil[i,j] = kamus[kata]
    
    return hasil

class ModelPhishingTransformer(nn.Module):
    """
    model transformer mini buat deteksi email phishing
    """
    def __init__(self, ukuran_kamus):
        super().__init__()
        self.embedding = nn.Embedding(ukuran_kamus, 16)
        self.attention = nn.MultiheadAttention(16, 4)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # ubah input jadi embedding
        x = self.embedding(x.long())
        x = x.permute(1, 0, 2)
        
        # hitung attention
        attn_output, attn_weights = self.attention(x, x, x)
        
        # klasifikasi
        out = attn_output.mean(dim=0)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out, attn_weights

def latih_model(model, data, label):
    """
    simulasi training model selama 10 epoch
    """
    optimizer = Adam(model.parameters())
    criterion = nn.BCELoss()
    
    print("mulai training...")
    for epoch in range(10):
        optimizer.zero_grad()
        output, _ = model(torch.tensor(data))
        loss = criterion(output.squeeze(), torch.tensor(label).float())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 2 == 0:
            print(f"epoch {epoch+1}/10 - loss: {loss.item():.4f}")

def gambar_heatmap(bobot_attention):
    """
    bikin visualisasi heatmap dari bobot attention
    """
    plt.figure(figsize=(10,8))
    plt.imshow(bobot_attention, cmap='hot')
    plt.colorbar()
    plt.title('heatmap bobot attention')
    plt.xlabel('posisi token')
    plt.ylabel('posisi token')

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    plt.savefig(os.path.join(assets_dir, 'heatmap_attention.png'))
    plt.close()

if __name__ == '__main__':
    # generate data
    print("membuat dataset...")
    email, label = buat_data_email()
    
    # preprocessing
    print("preprocessing data...")
    data = proses_teks(email)
    
    # bikin & latih model
    print("inisialisasi model...")
    model = ModelPhishingTransformer(51)  # 50 kata + 1 padding
    latih_model(model, data, label)
    
    # ambil bobot attention buat visualisasi
    print("membuat visualisasi...")
    with torch.no_grad():
        _, bobot = model(torch.tensor(data[:1]))
    gambar_heatmap(bobot[0].numpy())
    
    print("selesai! hasil visualisasi disimpan di heatmap_attention.png")
