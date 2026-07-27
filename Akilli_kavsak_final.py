import os
import time
import cv2
from ultralytics import YOLO

# ==========================================
# 1. MODEL VE VİDEO KONTROLÜ
# ==========================================
model_path = "best.pt"

if not os.path.exists(model_path):
    print(f"HATA: '{model_path}' dosyası bulunamadı!")
    exit()

print("Şampiyon model yüklendi! Fiziksel Yönleri Düzeltilmiş ITS Sistemi başlatılıyor...")
model = YOLO(model_path)

video_path = "istanbul_trafik.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"HATA: '{video_path}' açılamadı!")
    exit()

# ==========================================
# 2. STANDARTLAŞTIRILMIŞ SAYIM HAFIZASI
# ==========================================
crossed_ids = set()
last_counted_time = {}

counts = {
    "gidis": {"toplam": 0, "arac": 0, "kamyon_otobus": 0, "ambulans": 0, "itfaiye": 0, "polis": 0},
    "gelis": {"toplam": 0, "arac": 0, "kamyon_otobus": 0, "ambulans": 0, "itfaiye": 0, "polis": 0}
}

LINE_Y = 400

# 🔥 GÜVEN EŞİKLERİ
POLICE_CONF_THRESHOLD = 0.96
EMERGENCY_CONF_THRESHOLD = 0.85

# 🔥 KRONOMETRELER VE ALARM HAFIZASI
alarm_end_time = 0
active_alarm_type = ""

# =========================================================================
# 🔥 YOĞUNLUK DURUMU VE 5 SANİYELİK ZAMAN KİLİDİ
# =========================================================================
gidis_durum_metin = "AKIS NORMAL"
gidis_durum_renk = (0, 255, 0)
gidis_durum_kilit_zamani = 0

gelis_durum_metin = "AKIS NORMAL"
gelis_durum_renk = (0, 255, 0)
gelis_durum_kilit_zamani = 0
# =========================================================================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video analizi tamamlandı.")
        break

    frame = cv2.resize(frame, (1280, 720))
    MID_X = 640
    line_color = (0, 255, 255)
    current_time = time.time()

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)

    anlik_gidis_arac = 0
    anlik_gelis_arac = 0

    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        names = results[0].names

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
        else:
            ids = [0] * len(boxes)

        for box, track_id, cls_idx, conf_score in zip(boxes, ids, clss, confs):
            x1, y1, x2, y2 = map(int, box)
            cls_name = names[cls_idx]
            cls_lower = cls_name.lower()

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            is_amb = any(w in cls_lower for w in ["ambul", "ambulance"])
            is_fire = any(w in cls_lower for w in ["fire", "itfaiye", "firetruck"])
            is_pol = any(w in cls_lower for w in ["polis", "police"])
            is_heavy = any(
                w in cls_lower for w in ["kamyon", "kamyonet", "bus", "otobus", "truck", "buyukarac", "belediye"])

            # Akıllı filtreler
            if is_pol and conf_score < POLICE_CONF_THRESHOLD:
                is_pol = False
            if (is_amb or is_fire) and conf_score < EMERGENCY_CONF_THRESHOLD:
                is_amb = False
                is_fire = False

            # Sınıf isimlendirme ve renkler
            if is_amb:
                box_color = (0, 0, 255)
                cls_name = "AMBULANS"
            elif is_fire:
                box_color = (0, 140, 255)
                cls_name = "ITFAIYE"
            elif is_pol:
                box_color = (255, 0, 0)
                cls_name = "POLIS"
            elif is_heavy:
                box_color = (0, 255, 255)
                cls_name = "Agir Ticari"
            else:
                box_color = (0, 255, 0)
                cls_name = "Arac"

            # =========================================================================
            # 🔥 FİZİKSEL YÖN EŞLEŞTİRMESİ (SOL: GİDİŞ, SAĞ: GELİŞ) 🔥
            # =========================================================================
            if center_x < MID_X:
                anlik_gidis_arac += 1
            else:
                anlik_gelis_arac += 1

            # Çizim
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            label_text = f"ID:{track_id} {cls_name}" if track_id != 0 else f"{cls_name}"
            cv2.putText(frame, f"{label_text} (%{int(conf_score * 100)})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

            # 140 piksellik dev yakalama ağı ve doğru yön ataması
            if track_id != 0 and (LINE_Y - 70) < center_y < (LINE_Y + 70):
                if track_id not in last_counted_time or (current_time - last_counted_time[track_id]) > 4.0:
                    last_counted_time[track_id] = current_time
                    line_color = (0, 255, 0)

                    # Sol taraf GİDİŞ, Sağ taraf GELİŞ olarak düzeltildi!
                    direction = "gidis" if center_x < MID_X else "gelis"
                    counts[direction]["toplam"] += 1

                    if is_amb:
                        counts[direction]["ambulans"] += 1
                        active_alarm_type = "AMBULANS"
                        alarm_end_time = current_time + 5.0
                    elif is_fire:
                        counts[direction]["itfaiye"] += 1
                        active_alarm_type = "ITFAIYE"
                        alarm_end_time = current_time + 5.0
                    elif is_pol:
                        counts[direction]["polis"] += 1
                        active_alarm_type = "POLIS"
                        alarm_end_time = current_time + 5.0
                    elif is_heavy:
                        counts[direction]["kamyon_otobus"] += 1
                    else:
                        counts[direction]["arac"] += 1
            # =========================================================================

    # Çizgiler
    cv2.line(frame, (0, LINE_Y), (1280, LINE_Y), line_color, 3)
    cv2.line(frame, (MID_X, 0), (MID_X, 720), (0, 255, 255), 2)

    # =========================================================================
    # 5. ZAMAN KİLİTLİ YOĞUNLUK HESABI
    # =========================================================================
    if current_time >= gidis_durum_kilit_zamani:
        if gidis_durum_metin == "AKIS NORMAL" and anlik_gidis_arac >= 3:
            gidis_durum_metin = "YOGUN TRAFIK"
            gidis_durum_renk = (0, 0, 255)
            gidis_durum_kilit_zamani = current_time + 5.0
        elif gidis_durum_metin == "YOGUN TRAFIK" and anlik_gidis_arac <= 1:
            gidis_durum_metin = "AKIS NORMAL"
            gidis_durum_renk = (0, 255, 0)
            gidis_durum_kilit_zamani = current_time + 5.0

    if current_time >= gelis_durum_kilit_zamani:
        if gelis_durum_metin == "AKIS NORMAL" and anlik_gelis_arac >= 3:
            gelis_durum_metin = "YOGUN TRAFIK"
            gelis_durum_renk = (0, 0, 255)
            gelis_durum_kilit_zamani = current_time + 5.0
        elif gelis_durum_metin == "YOGUN TRAFIK" and anlik_gelis_arac <= 1:
            gelis_durum_metin = "AKIS NORMAL"
            gelis_durum_renk = (0, 255, 0)
            gelis_durum_kilit_zamani = current_time + 5.0

            # Karartmalı Arka Plan Kutuları
    cv2.rectangle(frame, (10, 10), (330, 200), (15, 15, 15), -1)
    cv2.rectangle(frame, (10, 10), (330, 200), (0, 255, 255), 1)

    cv2.rectangle(frame, (650, 10), (970, 200), (15, 15, 15), -1)
    cv2.rectangle(frame, (650, 10), (970, 200), (0, 255, 255), 1)

    # SOL PANO (GİDİŞ YÖNÜ İSTATİSTİKLERİ - Sol şeritteki yoğun trafik buraya yazılır)
    cv2.putText(frame, f"GIDIS TOPLAM: {counts['gidis']['toplam']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (0, 255, 0), 2)
    cv2.putText(frame, f"Durum          : {gidis_durum_metin}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                gidis_durum_renk, 2)
    cv2.putText(frame, f"Binek Arac     : {counts['gidis']['arac']}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    cv2.putText(frame, f"Kamyon/Otobus  : {counts['gidis']['kamyon_otobus']}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"AMBULANS       : {counts['gidis']['ambulans']}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 2)
    cv2.putText(frame, f"ITFAIYE        : {counts['gidis']['itfaiye']}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 140, 255), 2)
    cv2.putText(frame, f"POLIS          : {counts['gidis']['polis']}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 100, 100), 2)

    # SAĞ PANO (GELİŞ YÖNÜ İSTATİSTİKLERİ - Sağ şeritten gelenler buraya yazılır)
    cv2.putText(frame, f"GELIS TOPLAM: {counts['gelis']['toplam']}", (660, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (0, 255, 0), 2)
    cv2.putText(frame, f"Durum          : {gelis_durum_metin}", (660, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                gelis_durum_renk, 2)
    cv2.putText(frame, f"Binek Arac     : {counts['gelis']['arac']}", (660, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    cv2.putText(frame, f"Kamyon/Otobus  : {counts['gelis']['kamyon_otobus']}", (660, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"AMBULANS       : {counts['gelis']['ambulans']}", (660, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 2)
    cv2.putText(frame, f"ITFAIYE        : {counts['gelis']['itfaiye']}", (660, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 140, 255), 2)
    cv2.putText(frame, f"POLIS          : {counts['gelis']['polis']}", (660, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 100, 100), 2)

    # =========================================================================
    # 6. KRONOMETRELİ ÖNCELİK ALARMI BANNER'I
    # =========================================================================
    if current_time < alarm_end_time:
        cv2.rectangle(frame, (0, 0), (1280, 40), (0, 0, 255), -1)
        cv2.putText(frame,
                    f"ACIL DURUM ARACI ({active_alarm_type}) TESPIT EDILDI - TRAFIK ISIKLARI ONCELIKLENDIRILIYOR!",
                    (30, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    # =========================================================================

    cv2.imshow("ISBAK Akilli Kavsak - ITS Kontrol Merkezi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()