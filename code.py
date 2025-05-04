import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
from cv2 import dnn_superres

# Görsellik ayarları
BG_COLOR = "#e3f2fd"
BUTTON_COLOR = "#2196f3"
BUTTON_HOVER = "#1976d2"
TEXT_COLOR = "#0d47a1"


# --- İyileştirme Fonksiyonları ---

def sharpen_image(image):
    """Daha yumuşak ve doğal görünen keskinleştirme efekti"""
    kernel = np.array([[-1, -1, -1],
                       [-1, 9.5, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.addWeighted(image, 0.5, sharpened, 0.5, 0)


def denoise_image(image):
    """Gelişmiş gürültü azaltma algoritması"""
    return cv2.fastNlMeansDenoisingColored(image, None, h=8, hColor=8,
                                           templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image):
    """Akıllı kontrast artırma"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced


def adjust_saturation(image, saturation_factor=1.4):
    """Kontrollü doygunluk artırma"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, np.array([saturation_factor]))
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def beautify_face(image):
    """Doğal görünümlü yüz güzelleştirme"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img_blur = cv2.bilateralFilter(image, d=9, sigmaColor=100, sigmaSpace=100)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result = image.copy()
    for (x, y, w, h) in faces:
        face_roi = img_blur[y:y + h, x:x + w]
        result[y:y + h, x:x + w] = cv2.addWeighted(result[y:y + h, x:x + w], 0.3, face_roi, 0.7, 0)
    return result


def apply_hdr(image):
    """Doğal HDR efekti"""
    hdr = cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)
    return hdr


def adjust_brightness_contrast(image, brightness=5, contrast=1.1):
    """Parlaklık ve kontrast optimizasyonu"""
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 1:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image


def ai_super_resolution(image, scale=4):
    """Yüksek kaliteli süper çözünürlük"""
    sr = dnn_superres.DnnSuperResImpl_create()
    model_path = "EDSR_x4.pb"

    if not os.path.exists(model_path):
        messagebox.showerror("Hata", "EDSR model dosyası bulunamadı. 'EDSR_x4.pb' dosyasını indirip klasöre ekleyin!")
        return image

    try:
        sr.readModel(model_path)
        sr.setModel("edsr", scale)
        result = sr.upsample(image)
        result = cv2.detailEnhance(result, sigma_s=5, sigma_r=0.1)
        return result
    except Exception as e:
        messagebox.showerror("Hata", f"AI Süper Çözünürlük hatası: {str(e)}")
        return image


def get_adaptive_params(image):
    """Fotoğraf boyutuna göre adaptif parametreler"""
    h, w = image.shape[:2]
    size_factor = (h * w) / (1920 * 1080)  # Full HD'ye göre normalize

    return {
        'denoise_h': max(3, min(15, 7 * size_factor)),
        'sharpen_factor': max(1.0, min(2.0, 1.3 * size_factor)),
        'saturation': max(1.1, min(1.8, 1.4 * size_factor))
    }


# --- Ana İşlem Fonksiyonları ---

def process_photo(photo_path, output_folder, mode="Normal"):
    """Fotoğraf işleme fonksiyonu"""
    img = cv2.imread(photo_path)
    if img is None:
        messagebox.showerror("Hata", f"Fotoğraf bulunamadı: {photo_path}")
        return

    # Adaptif parametreler
    params = get_adaptive_params(img)

    # Temel iyileştirmeler
    img = adjust_brightness_contrast(img)
    img = denoise_image(img)

    if mode == "Normal":
        img = enhance_contrast(img)
        img = adjust_saturation(img, params['saturation'])
        img = sharpen_image(img)

    elif mode == "Yüksek Kalite":
        img = enhance_contrast(img)
        img = adjust_saturation(img, params['saturation'] * 1.1)
        for _ in range(2):
            img = sharpen_image(img)
        img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)

    elif mode == "HDR + Güzelleştirme":
        img = apply_hdr(img)
        img = beautify_face(img)
        img = adjust_saturation(img, params['saturation'] * 0.9)
        img = enhance_contrast(img)

    elif mode == "AI Süper Çözünürlük":
        img = ai_super_resolution(img)
        img = adjust_saturation(img, params['saturation'])
        img = enhance_contrast(img)

    # Yüksek kaliteli kayıt
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(photo_path)
    name, ext = os.path.splitext(filename)

    if ext.lower() in ['.jpg', '.jpeg']:
        output_path = os.path.join(output_folder, f"enhanced_{name}.jpg")
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    elif ext.lower() == '.png':
        output_path = os.path.join(output_folder, f"enhanced_{name}.png")
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    else:
        output_path = os.path.join(output_folder, f"enhanced_{filename}")
        cv2.imwrite(output_path, img)


def process_video(video_path, output_folder):
    """Video işleme fonksiyonu"""

    def process():
        nonlocal progress_window
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Hata", f"Video bulunamadı: {video_path}")
            progress_window.destroy()
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, "enhanced_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or cancel_flag[0]:
                break

            # Her frame için temel iyileştirmeler
            frame = adjust_brightness_contrast(frame)
            frame = denoise_image(frame)
            frame = enhance_contrast(frame)
            frame = adjust_saturation(frame, 1.3)

            out.write(frame)
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            progress_bar['value'] = progress
            percent_label.config(text=f"%{progress:.1f} tamamlandı")
            progress_window.update()

        cap.release()
        out.release()
        progress_window.destroy()

        if cancel_flag[0]:
            messagebox.showinfo("İptal Edildi", "Video işleme iptal edildi.")
            if os.path.exists(output_path):
                os.remove(output_path)
        else:
            messagebox.showinfo("Başarılı", f"Video iyileştirildi:\n{output_path}")

    # İlerleme penceresi
    progress_window = tk.Toplevel(window)
    progress_window.title("Video İşleniyor...")
    progress_window.geometry("400x150")
    progress_window.configure(bg=BG_COLOR)
    progress_window.resizable(False, False)

    # İlerleme çubuğu
    progress_bar = ttk.Progressbar(progress_window, length=350, mode='determinate')
    progress_bar.pack(pady=20)

    percent_label = tk.Label(progress_window, text="%0 tamamlandı",
                             bg=BG_COLOR, fg=TEXT_COLOR, font=('Helvetica', 10))
    percent_label.pack()

    cancel_flag = [False]

    def cancel_process():
        cancel_flag[0] = True
        cancel_button.config(state=tk.DISABLED)

    cancel_button = tk.Button(progress_window, text="İptal", bg="#f44336", fg="white",
                              activebackground="#d32f2f", command=cancel_process)
    cancel_button.pack(pady=10)

    # İşlemi thread'de başlat
    threading.Thread(target=process, daemon=True).start()


# --- Kullanıcı Arayüzü Fonksiyonları ---

def select_files_photo():
    """Fotoğraf seçme ve işleme"""
    filepaths = filedialog.askopenfilenames(
        title="Fotoğrafları Seçin",
        filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if not filepaths:
        return

    save_folder = filedialog.askdirectory(title="Kayıt Klasörünü Seçin")
    if not save_folder:
        return

    mode = quality_mode.get()

    # İşlem sayacı
    progress_window = tk.Toplevel(window)
    progress_window.title("Fotoğraflar İşleniyor...")
    progress_window.geometry("400x150")
    progress_window.configure(bg=BG_COLOR)
    progress_window.resizable(False, False)

    progress_label = tk.Label(progress_window, text=f"0/{len(filepaths)} fotoğraf işlendi",
                              bg=BG_COLOR, fg=TEXT_COLOR, font=('Helvetica', 10))
    progress_label.pack(pady=20)

    progress_bar = ttk.Progressbar(progress_window, length=350, maximum=len(filepaths),
                                   mode='determinate')
    progress_bar.pack()

    def process_images():
        for i, filepath in enumerate(filepaths, 1):
            if cancel_flag[0]:
                break
            process_photo(filepath, save_folder, mode=mode)
            progress_bar['value'] = i
            progress_label.config(text=f"{i}/{len(filepaths)} fotoğraf işlendi")
            progress_window.update()

        progress_window.destroy()
        if not cancel_flag[0]:
            messagebox.showinfo("Başarılı",
                                f"{len(filepaths)} fotoğraf iyileştirildi ve kaydedildi:\n{save_folder}")

    cancel_flag = [False]

    def cancel_process():
        cancel_flag[0] = True
        cancel_button.config(state=tk.DISABLED)

    cancel_button = tk.Button(progress_window, text="İptal", bg="#f44336", fg="white",
                              activebackground="#d32f2f", command=cancel_process)
    cancel_button.pack(pady=10)

    threading.Thread(target=process_images, daemon=True).start()


def select_file_video():
    """Video seçme ve işleme"""
    filepath = filedialog.askopenfilename(
        title="Video Dosyası Seçin",
        filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv *.flv")]
    )

    if not filepath:
        return

    save_folder = filedialog.askdirectory(title="Kayıt Klasörünü Seçin")
    if not save_folder:
        return

    process_video(filepath, save_folder)


def show_about():
    """Hakkında penceresi"""
    about_window = tk.Toplevel(window)
    about_window.title("Hakkında")
    about_window.geometry("400x300")
    about_window.configure(bg=BG_COLOR)
    about_window.resizable(False, False)

    title_label = tk.Label(about_window, text="Görüntü İyileştirici",
                           font=("Helvetica", 18, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
    title_label.pack(pady=10)

    version_label = tk.Label(about_window, text="Sürüm 2.0",
                             font=("Helvetica", 12), bg=BG_COLOR, fg=TEXT_COLOR)
    version_label.pack()

    desc_label = tk.Label(about_window,
                          text="Bu uygulama fotoğraf ve videolarınızı\nyapay zeka destekli algoritmalarla iyileştirir.",
                          font=("Helvetica", 11), bg=BG_COLOR, fg=TEXT_COLOR)
    desc_label.pack(pady=10)

    features_label = tk.Label(about_window, justify=tk.LEFT,
                              text="• Gelişmiş gürültü azaltma\n• Akıllı keskinleştirme\n• Yüz güzelleştirme\n• HDR efekti\n• Süper çözünürlük",
                              font=("Helvetica", 10), bg=BG_COLOR, fg=TEXT_COLOR)
    features_label.pack()

    team_label = tk.Label(about_window,
                          text="\nGeliştiriciler: 2220656621, 2230656820, 2220656061",
                          font=("Helvetica", 9), bg=BG_COLOR, fg="#78909c")
    team_label.pack(side=tk.BOTTOM, pady=10)


# --- Ana Pencere ---

window = tk.Tk()
window.title("✨ Profesyonel Görüntü İyileştirici ✨")
window.geometry("500x450")
window.configure(bg=BG_COLOR)
window.resizable(False, False)

# Başlık
title_label = tk.Label(window, text="Görüntü İyileştirici",
                       font=("Helvetica", 22, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
title_label.pack(pady=(20, 10))

# Mod Seçimi
mode_frame = tk.Frame(window, bg=BG_COLOR)
mode_frame.pack(pady=10)

mode_label = tk.Label(mode_frame, text="İşlem Modu:", bg=BG_COLOR, fg=TEXT_COLOR,
                      font=("Helvetica", 10))
mode_label.pack(side=tk.LEFT)

quality_mode = tk.StringVar(value="Normal")

mode_options = ["Normal", "Yüksek Kalite", "HDR + Güzelleştirme", "AI Süper Çözünürlük"]
mode_menu = ttk.Combobox(mode_frame, textvariable=quality_mode, values=mode_options,
                         state="readonly", width=20, font=("Helvetica", 10))
mode_menu.pack(side=tk.LEFT, padx=10)

# Butonlar
button_frame = tk.Frame(window, bg=BG_COLOR)
button_frame.pack(pady=20)

photo_button = tk.Button(button_frame, text="📷 Fotoğrafları İyileştir",
                         font=("Helvetica", 12), bg=BUTTON_COLOR, fg="white",
                         activebackground=BUTTON_HOVER, command=select_files_photo)
photo_button.pack(fill=tk.X, pady=5)

video_button = tk.Button(button_frame, text="🎥 Videoyu İyileştir",
                         font=("Helvetica", 12), bg=BUTTON_COLOR, fg="white",
                         activebackground=BUTTON_HOVER, command=select_file_video)
video_button.pack(fill=tk.X, pady=5)

# Hakkında Butonu
about_button = tk.Button(window, text="ℹ Hakkında", font=("Helvetica", 10),
                         bg="#607d8b", fg="white", activebackground="#455a64",
                         command=show_about)
about_button.pack(pady=10)

# Footer
footer_label = tk.Label(window, text="© 2023 Görüntü İyileştirici | Tüm Hakları Saklıdır",
                        font=("Helvetica", 8), bg=BG_COLOR, fg="#78909c")
footer_label.pack(side=tk.BOTTOM, pady=5)

window.mainloop()
