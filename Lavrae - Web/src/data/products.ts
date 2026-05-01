import p1 from "@/assets/p1.jpg";
import p2 from "@/assets/p2.jpg";
import p3 from "@/assets/p3.jpg";
import p4 from "@/assets/p4.jpg";
import p5 from "@/assets/p5.jpg";
import p6 from "@/assets/p6.jpg";
import p7 from "@/assets/p7.jpg";
import p8 from "@/assets/p8.jpg";

export type ProductTag = "New" | "Best Seller" | "Limited";

export interface Product {
  id: string;
  slug: string;
  name: string;
  price: number;
  category: "tops" | "kurtis" | "dresses" | "sets";
  tags: ProductTag[];
  images: string[];
  description: string;
  fabric: string;
}

export const categories = [
  { slug: "tops", name: "Tops", image: p1, blurb: "Tailored silhouettes" },
  { slug: "kurtis", name: "Kurtis", image: p2, blurb: "Heritage, reimagined" },
  { slug: "dresses", name: "Dresses", image: p3, blurb: "Effortless elegance" },
  { slug: "sets", name: "Sets", image: p4, blurb: "Co-ordinated ease" },
] as const;

export const products: Product[] = [
  {
    id: "1", slug: "ivory-silk-blouse", name: "Ivory Silk Blouse", price: 4800,
    category: "tops", tags: ["Best Seller"], images: [p1, p5],
    description: "A signature blouse in pure mulberry silk with hand-finished cuffs.",
    fabric: "100% Mulberry Silk",
  },
  {
    id: "2", slug: "noor-embroidered-kurti", name: "Noor Embroidered Kurti", price: 6200,
    category: "kurtis", tags: ["New"], images: [p2, p8],
    description: "Hand-embroidered yoke on soft cotton-silk, tailored for everyday grace.",
    fabric: "Cotton Silk",
  },
  {
    id: "3", slug: "cream-bow-midi-dress", name: "Cream Bow Midi Dress", price: 7400,
    category: "dresses", tags: ["Best Seller"], images: [p3, p7],
    description: "A fluid midi with a sash bow, cut from washed satin.",
    fabric: "Washed Satin",
  },
  {
    id: "4", slug: "atelier-tailored-set", name: "Atelier Tailored Set", price: 9800,
    category: "sets", tags: ["New", "Limited"], images: [p4, p1],
    description: "A longline blazer paired with high-waist trousers in ivory wool blend.",
    fabric: "Wool Blend",
  },
  {
    id: "5", slug: "maroon-zardozi-blouse", name: "Maroon Zardozi Blouse", price: 8400,
    category: "tops", tags: ["Limited"], images: [p5, p2],
    description: "Deep maroon crepe with hand zardozi — a quiet nod to royalty.",
    fabric: "Silk Crepe",
  },
  {
    id: "6", slug: "linen-kurta-tunic", name: "Linen Kurta Tunic", price: 4200,
    category: "kurtis", tags: ["Best Seller"], images: [p6, p8],
    description: "Breathable linen, mandarin collar, made for slow days.",
    fabric: "Pure Linen",
  },
  {
    id: "7", slug: "garden-trench-dress", name: "Garden Trench Dress", price: 8900,
    category: "dresses", tags: ["New"], images: [p7, p3],
    description: "A trench-inspired silhouette in flowing camel crepe.",
    fabric: "Crepe",
  },
  {
    id: "8", slug: "noor-co-ord-set", name: "Noor Co-ord Set", price: 7600,
    category: "sets", tags: ["New"], images: [p8, p4],
    description: "Embroidered kurta with matching straight pants in champagne ivory.",
    fabric: "Cotton Silk",
  },
];

export const formatINR = (n: number) =>
  new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(n);
