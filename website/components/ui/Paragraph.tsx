import clsx from "clsx";

interface ParagraphProps {
  children: React.ReactNode;
  className?: string;
  size?: "sm" | "base" | "lg";
}

export default function Paragraph({
  children,
  className,
  size = "base",
}: ParagraphProps) {
  const sizes = {
    sm: "text-sm",
    base: "text-base md:text-lg",
    lg: "text-lg md:text-xl",
  };

  return (
    <p
      className={clsx(
        "font-[family-name:var(--font-body)]",
        sizes[size],
        "text-starlight leading-relaxed",
        className
      )}
    >
      {children}
    </p>
  );
}
