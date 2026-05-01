import { useMemo, useState } from "react";
import Layout from "@/components/layout/Layout";
import ProductCard from "@/components/ProductCard";
import { products, categories } from "@/data/products";
import { Search } from "lucide-react";

interface Props {
  title: string;
  eyebrow?: string;
  description?: string;
  filterCategory?: string;
  filterTag?: string;
}

const Shop = ({ title, eyebrow, description, filterCategory, filterTag }: Props) => {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"featured" | "asc" | "desc">("featured");
  const [activeCat, setActiveCat] = useState<string>(filterCategory ?? "all");

  const list = useMemo(() => {
    let r = products.slice();
    if (filterTag) r = r.filter(p => p.tags.includes(filterTag as any));
    if (filterCategory) r = r.filter(p => p.category === filterCategory);
    else if (activeCat !== "all") r = r.filter(p => p.category === activeCat);
    if (query.trim()) {
      const q = query.toLowerCase();
      r = r.filter(p => p.name.toLowerCase().includes(q) || p.fabric.toLowerCase().includes(q));
    }
    if (sort === "asc") r.sort((a, b) => a.price - b.price);
    if (sort === "desc") r.sort((a, b) => b.price - a.price);
    return r;
  }, [query, sort, activeCat, filterCategory, filterTag]);

  return (
    <Layout>
      <section className="pt-32 md:pt-40 pb-10 container text-center">
        {eyebrow && <p className="text-xs uppercase tracking-luxury text-accent mb-3">{eyebrow}</p>}
        <h1 className="font-serif text-4xl md:text-6xl">{title}</h1>
        {description && <p className="mt-4 text-muted-foreground max-w-xl mx-auto">{description}</p>}
      </section>

      <section className="container">
        {/* Filter bar */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 py-6 border-y border-border">
          {!filterCategory ? (
            <div className="flex gap-1 overflow-x-auto scrollbar-none">
              {[{ slug: "all", name: "All" }, ...categories].map(c => (
                <button
                  key={c.slug}
                  onClick={() => setActiveCat(c.slug)}
                  className={`px-4 py-2 text-xs uppercase tracking-luxury whitespace-nowrap transition-colors ${
                    activeCat === c.slug ? "text-accent border-b border-accent" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {c.name}
                </button>
              ))}
            </div>
          ) : <div />}

          <div className="flex items-center gap-3">
            <div className="flex items-center border border-border focus-within:border-accent transition-colors">
              <Search className="h-3.5 w-3.5 ml-3 text-muted-foreground" />
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Search"
                className="bg-transparent px-2 py-2 text-sm outline-none w-32 md:w-44"
              />
            </div>
            <select
              value={sort}
              onChange={e => setSort(e.target.value as any)}
              className="bg-transparent border border-border px-3 py-2 text-xs uppercase tracking-luxury outline-none focus:border-accent"
            >
              <option value="featured">Featured</option>
              <option value="asc">Price: Low to High</option>
              <option value="desc">Price: High to Low</option>
            </select>
          </div>
        </div>

        {/* Grid */}
        <div className="py-12">
          {list.length === 0 ? (
            <p className="text-center text-muted-foreground py-20">No pieces match your search.</p>
          ) : (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8">
              {list.map((p, i) => <ProductCard key={p.id} product={p} index={i} />)}
            </div>
          )}
        </div>
      </section>
    </Layout>
  );
};

export default Shop;
