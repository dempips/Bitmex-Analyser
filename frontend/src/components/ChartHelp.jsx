import React from "react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

/**
 * Small question-mark icon for chart cards. On hover/click shows plain-English explanation.
 */
export function ChartHelp({ title, children, className }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className={cn("h-7 w-7 rounded-full text-muted-foreground hover:text-foreground", className)}
          aria-label="Chart help"
        >
          <span className="text-sm font-medium">?</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 text-sm space-y-1" side="left" align="end">
        {title && <p className="font-medium">{title}</p>}
        {children}
      </PopoverContent>
    </Popover>
  );
}
