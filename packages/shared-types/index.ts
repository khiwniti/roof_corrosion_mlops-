import { z } from "zod";

// ── Quote job ──────────────────────────────────────────────
export const QuoteRequestSchema = z.object({
  address: z.string().optional(),
  lat: z.number().optional(),
  lng: z.number().optional(),
  polygonGeojson: z.record(z.unknown()).optional(),
});
export type QuoteRequest = z.infer<typeof QuoteRequestSchema>;

export const QuoteResponseSchema = z.object({
  jobId: z.string(),
  status: z.enum(["queued", "processing", "completed", "failed"]),
  message: z.string(),
});
export type QuoteResponse = z.infer<typeof QuoteResponseSchema>;

// ── Quote result ───────────────────────────────────────────
export const CorrosionAssessmentSchema = z.object({
  roofAreaM2: z.number(),
  corrodedAreaM2: z.number(),
  corrosionPercent: z.number(),
  severity: z.enum(["none", "light", "moderate", "severe"]),
  confidence: z.number().min(0).max(1),
  tileCaptureDate: z.string().datetime().optional(),
});
export type CorrosionAssessment = z.infer<typeof CorrosionAssessmentSchema>;

export const QuoteLineItemSchema = z.object({
  description: z.string(),
  quantity: z.number(),
  unit: z.string(),
  unitPrice: z.number(),
  total: z.number(),
});
export type QuoteLineItem = z.infer<typeof QuoteLineItemSchema>;

export const QuoteResultSchema = z.object({
  jobId: z.string(),
  assessment: CorrosionAssessmentSchema,
  lineItems: z.array(QuoteLineItemSchema),
  totalAmount: z.number(),
  currency: z.string().default("USD"),
  overlayImageUrl: z.string().url().optional(),
  requiresHumanReview: z.boolean(),
});
export type QuoteResult = z.infer<typeof QuoteResultSchema>;

// ── Feedback ───────────────────────────────────────────────
export const FeedbackRequestSchema = z.object({
  jobId: z.string(),
  correct: z.boolean(),
  notes: z.string().optional(),
});
export type FeedbackRequest = z.infer<typeof FeedbackRequestSchema>;
