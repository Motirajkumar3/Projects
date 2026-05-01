import { useParams, Navigate } from "react-router-dom";
import Shop from "./Shop";
import { categories } from "@/data/products";

const Category = () => {
  const { slug } = useParams();
  const cat = categories.find(c => c.slug === slug);
  if (!cat) return <Navigate to="/shop" replace />;
  return (
    <Shop
      title={cat.name}
      eyebrow="Category"
      description={cat.blurb}
      filterCategory={cat.slug}
    />
  );
};

export default Category;
