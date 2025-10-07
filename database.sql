-- Kapasite parametreleri için örnek şema ve örnek veri.
-- Üretimde kendi şemanızı kullanın. Uygulama yalnızca SELECT çıktısını okur.

-- Örnek tablo:
-- CREATE TABLE depo_kapasite (
--   depo TEXT NOT NULL,
--   urun TEXT,
--   kapasite NUMERIC NOT NULL
-- );

-- Örnek veri:
-- INSERT INTO depo_kapasite (depo, urun, kapasite) VALUES
-- ('Bandirma', 'Boraks', 120000),
-- ('Kirka', 'Kalsine Tinkal', 90000),
-- ('Emet', 'Etibor', 60000),
-- ('Bigadic', NULL, 75000);

-- Uygulama, script içindeki SON SELECT çıktısını kullanır:
SELECT depo, urun, kapasite
FROM depo_kapasite;
