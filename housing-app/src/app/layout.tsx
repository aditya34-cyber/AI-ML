import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "California Housing Price Predictor | ML-Powered",
  description:
    "Interactive map to predict California housing prices using machine learning. Click anywhere on the map to get instant price predictions based on location and housing data.",
  keywords: [
    "California housing",
    "price prediction",
    "machine learning",
    "real estate",
    "house prices",
  ],
  authors: [{ name: "AI-ML Project" }],
  openGraph: {
    title: "California Housing Price Predictor",
    description: "ML-powered interactive map for California housing price predictions",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
