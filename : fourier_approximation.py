import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import warnings

warnings.filterwarnings('ignore')

# 1. Hedef Fonksiyon: f(x) = x^2 + x
def target_func(x):
    return x**2 + x

# 2. Fourier Serisi Katsayıları ve Seri Değerlerini Hesaplayan Fonksiyon
def calculate_fourier_series(x_vals, N):
    """
    [-pi, pi] aralığında f(x) fonksiyonunun ilk N terimli 
    Fourier serisi açılımını sayısal integrasyon (quad) ile hesaplar.
    """
    # a0 katsayısı: (1 / pi) * integral(f(x) dx, -pi, pi)
    integral_a0, _ = quad(target_func, -np.pi, np.pi, limit=200)
    a0 = integral_a0 / np.pi
    
    # Seriyi a0 / 2 ile başlat
    series = np.full_like(x_vals, a0 / 2.0)
    
    # an ve bn terimlerini hesapla ve seriye ekle
    for n in range(1, N + 1):
        def integrand_an(x):
            return target_func(x) * np.cos(n * x)
        
        def integrand_bn(x):
            return target_func(x) * np.sin(n * x)
        
        integral_an, _ = quad(integrand_an, -np.pi, np.pi, limit=200)
        integral_bn, _ = quad(integrand_bn, -np.pi, np.pi, limit=200)
        
        an = integral_an / np.pi
        bn = integral_bn / np.pi
        
        series += an * np.cos(n * x_vals) + bn * np.sin(n * x_vals)
        
    return series

def main():
    print("Hesaplanıyor, lütfen bekleyiniz...")
    
    # [-pi, pi] aralığında x değerleri
    x = np.linspace(-np.pi, np.pi, 1000)
    y_real = target_func(x)
    
    # Karşılaştırılacak N (terim sayısı) değerleri
    N_list = [2, 5, 10, 50, 100]
    
    # 2x3'lük subplot ızgarası oluştur
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(r"Fourier Serisi Yakınsaması: $f(x) = x^2 + x$", fontsize=16, weight='bold')
    
    axes_flat = axes.flatten()
    
    # Her N değeri için grafiği çiz
    for idx, N in enumerate(N_list):
        ax = axes_flat[idx]
        y_approx = calculate_fourier_series(x, N)
        
        ax.plot(x, y_real, 'g--', linewidth=2, label="Orijinal f(x)")
        ax.plot(x, y_approx, 'b-', linewidth=1.5, label=f"Fourier (N={N})")
        
        ax.set_title(f"Terim Sayısı N = {N}", fontsize=12)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="upper left")
        
    # Son kutuyu yorum alanı olarak kullan (belgedeki gibi)
    ax_last = axes_flat[-1]
    ax_last.axis('off')
    yorum_metni = (
        "YORUM:\n\n"
        "• N (terim sayısı) arttıkça\n"
        "  Fourier serisi orijinal fonksiyona\n"
        "  daha iyi yaklaşmaktadır.\n\n"
        "• Sayısal integral hata payı\n"
        "  N büyüdükçe küçülmekte ve\n"
        "  kare dalgalanmalar sönümlenmektedir."
    )
    ax_last.text(0.1, 0.4, yorum_metni, fontsize=13, weight='bold', color='#1a365d', va='center')
    
    plt.tight_layout()
    plt.savefig("fourier_series_approximation.png", dpi=300)
    print("Grafik 'fourier_series_approximation.png' olarak başarıyla kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()
