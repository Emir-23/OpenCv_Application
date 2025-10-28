using System;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace HumanDetection
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Console.WriteLine("İnsan Algılama Sistemi Başlatılıyor...");
            Console.WriteLine("ESC tuşuna basarak programı sonlandırabilirsiniz.");
            Console.WriteLine("Press any key to start...");
            Console.ReadKey();
            
            DetectHuman();
        }

        static void DetectHuman()
        {
            // Webcam'i aç
            Console.WriteLine("\nKamera açılıyor...");
            
            VideoCapture capture = null;
            
            // Farklı kamera indekslerini dene
            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine($"Kamera {i} deneniyor...");
                capture = new VideoCapture(i);
                
                if (capture.IsOpened)
                {
                    Console.WriteLine($"✅ Kamera {i} başarıyla açıldı.");
                    break;
                }
                else
                {
                    capture.Dispose();
                    capture = null;
                }
            }
            
            if (capture == null || !capture.IsOpened)
            {
                Console.WriteLine("❌ Hiçbir kamera açılamadı!");
                Console.WriteLine("Lütfen kamera bağlantınızı kontrol edin.");
                Console.WriteLine("Bir tuşa basın...");
                Console.ReadKey();
                return;
            }

            // HOG (Histogram of Oriented Gradients) ile insan algılama
            Mat frame = new Mat();
            Mat grayFrame = new Mat();
            
            Console.WriteLine("HOG insan algılama modülü yükleniyor...");
            HOGDescriptor hog = new HOGDescriptor();
            // HOG parametreleri - insan tespiti için
            hog.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());
            Console.WriteLine("✅ Sistem hazır!");

            Console.WriteLine("\nİnsan algılama başladı. ESC tuşu ile çıkış yapabilirsiniz.\n");

            int frameCount = 0;
            DateTime lastDetectionTime = DateTime.Now;
            bool humanDetected = false;

            while (true)
            {
                capture.Read(frame);
                
                if (frame.IsEmpty)
                    break;

                frameCount++;
                
                // Her 5 frame'de bir kontrol yap (performans için)
                if (frameCount % 5 == 0)
                {
                    // Vücut algılama (HOG)
                    Rectangle[] bodies;
                    
                    try
                    {
                        // HOG algoritması MCvObjectDetection[] döndürür, Rectangle[]'e çeviriyoruz
                        MCvObjectDetection[] detections = hog.DetectMultiScale(
                            frame,
                            winStride: new Size(8, 8),
                            padding: new Size(32, 32),
                            scale: 1.05);
                        
                        // MCvObjectDetection'dan Rectangle çıkar
                        bodies = detections.Select(d => d.Rect).ToArray();
                    }
                    catch
                    {
                        bodies = Array.Empty<Rectangle>();
                    }

                    // İnsan tespiti yap
                    bool currentDetection = bodies.Length > 0;
                    
                    // Sonuçları göster
                    Console.Write($"\r[Frame: {frameCount}] ");

                    if (bodies.Length > 0)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.Write("✓ İNSAN TESPİT EDİLDİ ");
                        Console.ResetColor();
                        
                        Console.Write($"| İnsan sayısı: {bodies.Length}");
                        
                        humanDetected = true;
                        lastDetectionTime = DateTime.Now;

                        // Tespit edilen vücutları çiz
                        foreach (var body in bodies)
                        {
                            CvInvoke.Rectangle(frame, body, new MCvScalar(0, 255, 0), 2);
                            CvInvoke.PutText(frame, "INSAN", 
                                new Point(body.X, body.Y - 10), 
                                FontFace.HersheySimplex, 0.8, 
                                new MCvScalar(0, 255, 0), 2);
                        }

                        // İnsan tespit edildi - mesaj göster
                        Console.WriteLine("\n\n");
                        Console.WriteLine("╔════════════════════════════════════════╗");
                        Console.WriteLine("║        İNSAN TESPİT EDİLDİ! ✓        ║");
                        Console.WriteLine("╚════════════════════════════════════════╝");
                        Console.WriteLine($"\nTespit edilen insan sayısı: {bodies.Length}");
                        Console.WriteLine($"\nToplam frame sayısı: {frameCount}");
                        Console.WriteLine("\nDevam ediliyor... (ESC ile çıkabilirsiniz)");
                    }
                    else
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        Console.Write("✗ İnsan tespit edilemedi");
                        Console.ResetColor();
                        humanDetected = false;
                    }
                }

                // Ekranda bilgi göster
                string statusText = humanDetected ? "INSAN VAR" : "INSAN YOK";
                MCvScalar statusColor = humanDetected ? new MCvScalar(0, 255, 0) : new MCvScalar(0, 0, 255);
                
                CvInvoke.Rectangle(frame, new Rectangle(10, 10, 200, 60), new MCvScalar(0, 0, 0), -1);
                CvInvoke.PutText(frame, statusText, new Point(20, 40), 
                    FontFace.HersheySimplex, 1.0, statusColor, 2);
                CvInvoke.PutText(frame, $"Frame: {frameCount}", new Point(20, 70), 
                    FontFace.HersheySimplex, 0.6, new MCvScalar(255, 255, 255), 1);

                // Frame'i göster
                CvInvoke.Imshow("Insan Algilama", frame);

                // ESC tuşuna basıldığında çık
                if (CvInvoke.WaitKey(1) == 27) // ESC tuşu
                {
                    Console.WriteLine("\n\nProgram sonlandırılıyor...");
                    capture.Dispose();
                    CvInvoke.DestroyAllWindows();
                    break;
                }
            }
        }
    }
}
