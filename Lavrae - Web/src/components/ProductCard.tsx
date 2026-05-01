import { Link } from "react-router-dom";
import { Product, formatINR } from "@/data/products";

const tagStyle = (tag: string) =>
  tag === "Best Seller"
    ? "bg-accent text-accent-foreground"
    : tag === "New"
    ? "bg-foreground text-background"
    : "bg-secondary text-foreground border border-border";

const ProductCard = ({ product, index = 0 }: { product: Product; index?: number }) => (
  <Link
    to={`/product/${product.slug}`}
    className="group block animate-fade-up"
    style={{ animationDelay: `${index * 60}ms` }}
  >
    <div className="relative aspect-[4/5] bg-secondary/50 hover-zoom rounded-sm">
      <img
        src={product.images[0]}
        alt={product.name}
        loading="lazy"
        width={900}
        height={1100}
        className="absolute inset-0 w-full h-full object-cover"
      />
      {product.images[1] && (
        <img
          src={product.images[1]}
          alt=""
          loading="lazy"
          aria-hidden
          className="absolute inset-0 w-full h-full object-cover opacity-0 group-hover:opacity-100 transition-opacity duration-700"
        />
      )}
      {product.tags[0] && (
        <span className={`absolute top-3 left-3 text-[10px] uppercase tracking-luxury px-2.5 py-1 ${tagStyle(product.tags[0])}`}>
          {product.tags[0]}
        </span>
      )}
    </div>
    <div className="pt-4 flex justify-between items-baseline gap-3">
      <h3 className="font-serif text-lg leading-tight group-hover:text-accent transition-colors">
        {product.name}
      </h3>
      <p className="text-sm text-muted-foreground whitespace-nowrap">{formatINR(product.price)}</p>
    </div>
  </Link>
);

export default ProductCard;
