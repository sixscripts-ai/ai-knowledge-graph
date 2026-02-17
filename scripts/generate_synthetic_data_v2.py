#!/usr/bin/env python3
"""
ICT Synthetic Training Data Generator — V2 (Gap Filler)
=========================================================
Generates high-quality Q&A pairs for the 159 concepts NOT covered
by the original 24-concept generator.

Sources knowledge from:
  - terminology.yaml definitions (train-ict + Antigravity)
  - concept_relationships.yaml 
  - ict_graph_final.json triples
  - rapid_fire_drill.md Q&A patterns
  - concept .md files across repos

Target: 300+ new examples covering Tier 1 & Tier 2 gaps.
"""
import json
import random
from pathlib import Path

OUTPUT_DIR = Path("finetune_data_v4")

SYSTEM_PROMPT = "You are VEX, an expert ICT (Inner Circle Trader) trading assistant. You help traders understand ICT methodology with clear, practical explanations."

# ============================================================
# GAP-FILLING CONCEPTS — organized by category
# Each has definition + Q&A pairs sourced from all 3 repos
# ============================================================

CONCEPTS = [
    # ══════════════════════════════════════════════════════════
    #  PRICE ACTION — Tier 1 gaps
    # ══════════════════════════════════════════════════════════
    {
        "name": "Inversion Fair Value Gap",
        "alias": ["IFVG"],
        "qa_pairs": [
            {
                "q": "What is an Inversion Fair Value Gap?",
                "a": "An Inversion Fair Value Gap (IFVG) is a failed FVG that undergoes a state change — it flips from its original role and acts as support or resistance from the opposite side. For example, a bullish FVG that was supposed to act as support but gets traded through becomes a bearish IFVG that now acts as resistance. This state change is significant because it tells you that institutional commitment has shifted. When price returns to test the IFVG from the other side, expect a reaction. IFVGs are particularly powerful when they align with a Market Structure Shift, confirming that the market has changed direction."
            },
            {
                "q": "How do I identify and trade an Inversion FVG?",
                "a": "To identify an IFVG, first find a standard FVG that has been completely traded through — price closed beyond it with displacement. That FVG has now 'inverted.' Mark the zone. When price retraces back to this inverted zone, it should act as the opposite role — a failed bullish FVG becomes bearish resistance, a failed bearish FVG becomes bullish support. Enter when price reaches the IFVG zone and shows a reaction (wick rejection, lower timeframe MSS). Place your stop beyond the IFVG extreme. Target the next liquidity pool on the other side. IFVGs are higher probability when they coincide with a Breaker Block at the same level."
            },
        ]
    },
    {
        "name": "Bullish Order Block",
        "alias": ["Bullish OB"],
        "qa_pairs": [
            {
                "q": "What specifically makes a Bullish Order Block?",
                "a": "A Bullish Order Block is the last bearish (down-close) candle before a strong bullish displacement move. It represents where institutions placed buy orders that initiated the rally. The key requirements: the candle must be bearish (close below open), it must be followed by displacement (large bullish candles breaking structure), and ideally that displacement creates a Fair Value Gap. The tradeable zone is the body of that bearish candle — specifically from its open down to its low. When price retraces to this zone, institutions' unfilled buy orders are still there, creating support. The most reliable Bullish OBs are found in discount zones below equilibrium."
            },
        ]
    },
    {
        "name": "Bearish Order Block",
        "alias": ["Bearish OB"],
        "qa_pairs": [
            {
                "q": "What specifically makes a Bearish Order Block?",
                "a": "A Bearish Order Block is the last bullish (up-close) candle before a strong bearish displacement move. It marks where institutions placed sell orders that initiated the sell-off. The candle must be bullish (close above open), followed by strong bearish displacement that breaks structure and ideally creates Fair Value Gaps. The tradeable zone is the body of that bullish candle — from its open up to its high. When price retraces up to this zone, expect resistance as institutional sell orders are still resting there. The most reliable Bearish OBs are in premium zones above equilibrium. Always validate with displacement — no displacement means no valid OB."
            },
        ]
    },
    {
        "name": "Balanced Price Range",
        "alias": ["BPR"],
        "qa_pairs": [
            {
                "q": "What is a Balanced Price Range in ICT?",
                "a": "A Balanced Price Range (BPR) forms when price aggressively moves through a swing high or low, then immediately returns inside the range. It creates a zone where both bullish and bearish FVGs overlap — essentially price traded through both sides of a level. The overlapping area of these opposing FVGs is the BPR. This zone is significant because it represents a truly 'balanced' area where both buyers and sellers have transacted. The midpoint of the BPR acts as a strong reaction level. BPRs are useful for re-entry when you missed the initial move — they give you a second chance to enter at a fair price after a liquidity sweep."
            },
            {
                "q": "How do I trade a Balanced Price Range?",
                "a": "To trade a BPR: First, identify a swing high or low that was swept (liquidity grabbed). Then look for the area where the bullish and bearish FVGs created during the sweep overlap — that overlap zone is your BPR. When price returns to the BPR midpoint, it often finds a reaction. For a bullish BPR trade: after a sell-side liquidity sweep, mark where the opposing FVGs overlap, then buy when price retraces to the BPR midpoint with your stop below the sweep low. For bearish: after a buy-side sweep, sell at the BPR midpoint with stop above the sweep high. The BPR works best during Kill Zones and when aligned with HTF bias."
            },
        ]
    },
    {
        "name": "Volume Imbalance",
        "alias": ["VI"],
        "qa_pairs": [
            {
                "q": "What is a Volume Imbalance in ICT?",
                "a": "A Volume Imbalance (VI) is a gap between two consecutive candles where the close of one candle and the open of the next don't overlap — there's a visible price gap between them. Unlike a Fair Value Gap which is a 3-candle pattern, a Volume Imbalance is a 2-candle pattern. It represents a moment where the market moved so fast between candle closes that no transactions occurred at those prices. Volume Imbalances act similarly to FVGs — price tends to return to fill them. They're particularly common around news events and session opens. In the PD Array hierarchy, they're treated as secondary imbalances that price is drawn to rebalance."
            },
        ]
    },
    {
        "name": "Liquidity Void",
        "alias": ["LV"],
        "qa_pairs": [
            {
                "q": "What is a Liquidity Void and how does it differ from an FVG?",
                "a": "A Liquidity Void is a long-bodied candle with minimal wicks, representing an extreme one-sided price movement where virtually no opposing transactions occurred. Unlike an FVG (which is a gap between wicks of candles 1 and 3 in a 3-candle pattern), a Liquidity Void is about the candle itself — its body is so large and wicks so small that it shows complete institutional dominance. The midpoint of a Liquidity Void often acts as a reaction level when price returns. Liquidity Voids confirm displacement events and are often found within institutional legs. They rank below Order Blocks and FVGs in the PD Array Matrix but serve as important confirmation tools."
            },
        ]
    },
    {
        "name": "New Week Opening Gap",
        "alias": ["NWOG"],
        "qa_pairs": [
            {
                "q": "What is a New Week Opening Gap?",
                "a": "A New Week Opening Gap (NWOG) is the price difference between Friday's close and Sunday/Monday's open. This gap represents institutional positioning that occurred over the weekend when retail traders couldn't participate. NWOGs act as significant reference levels throughout the trading week — they function like magnets that price is drawn toward. The midpoint of a NWOG (the 50% level between the gap's high and low) is especially important as a support/resistance level. When price trades through a NWOG, it often signals strong institutional commitment in that direction. NWOGs rank in the PD Array Matrix after Mitigation Blocks."
            },
        ]
    },
    {
        "name": "New Day Opening Gap",
        "alias": ["NDOG"],
        "qa_pairs": [
            {
                "q": "What is a New Day Opening Gap?",
                "a": "A New Day Opening Gap (NDOG) is the price gap between one session's close and the next session's open. Similar to NWOGs but on a daily timeframe, NDOGs reveal where institutional orders were placed between sessions. The gap zone acts as a reference point that price may return to fill during the trading day. NDOGs are particularly relevant during the London and NY sessions — an unfilled NDOG becomes a draw on liquidity. Like all gaps in ICT methodology, the 50% midpoint of the NDOG is the key level. Price filling an NDOG before continuing in its original direction is a common institutional pattern and can provide low-risk entry opportunities."
            },
        ]
    },
    {
        "name": "Consequent Encroachment",
        "alias": ["CE"],
        "qa_pairs": [
            {
                "q": "What is Consequent Encroachment?",
                "a": "Consequent Encroachment (CE) is the exact 50% midpoint of a Fair Value Gap. The formula is simple: (FVG High + FVG Low) / 2. This level is significant because it represents the mathematical center of the imbalance — half of the inefficiency has been 'encroached' upon. In practice, CE acts as a precision entry point within an FVG. Rather than entering at the top or bottom of the gap, entering at the CE gives you a tighter stop and better risk:reward. If a candle body closes beyond the CE level, it signals that the FVG may fail — the imbalance is being fully rebalanced. CE is used for surgical entries in Silver Bullet and scalping setups."
            },
        ]
    },
    {
        "name": "Mean Threshold",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the Mean Threshold in ICT?",
                "a": "The Mean Threshold is the midpoint of an Order Block — calculated as (Open + Close) / 2 of the OB candle. It serves as a precision entry level within the Order Block zone. Instead of entering at the first touch of the OB, waiting for price to reach the Mean Threshold gives you a deeper entry with tighter risk. Smart money often respects this level because it represents the average price at which institutional orders were filled. If price trades through the Mean Threshold and closes beyond the OB body on a candle basis, the OB is likely invalidated. The Mean Threshold is to Order Blocks what Consequent Encroachment is to Fair Value Gaps — a 50% precision level."
            },
        ]
    },
    {
        "name": "Premium and Discount Zones",
        "alias": ["Premium", "Discount"],
        "qa_pairs": [
            {
                "q": "What are Premium and Discount zones in ICT?",
                "a": "Premium and Discount zones divide any dealing range into expensive and cheap areas using the 50% equilibrium level. The formula: Equilibrium = (Swing High + Swing Low) / 2. Everything above equilibrium is Premium (expensive) — you should only look for sells here. Everything below is Discount (cheap) — you should only look for buys here. This is fundamental to ICT: buying in premium or selling in discount goes against institutional logic. Smart money buys cheap and sells expensive. When looking for Order Blocks or FVGs to trade, always check if they're in the correct zone — a bullish OB in premium is suspect, a bullish OB in discount is high-probability."
            },
            {
                "q": "How does Equilibrium work with Premium and Discount?",
                "a": "Equilibrium is the 50% midpoint of any swing range — the dividing line between Premium and Discount. It's calculated as (Swing High + Swing Low) / 2. Equilibrium acts as a key reference: price often consolidates around equilibrium before making its next institutional move. In a bullish scenario, you want to buy when price is below equilibrium (in discount) and target liquidity above equilbrium (in premium). In a bearish scenario, sell in premium and target liquidity in discount. Equilibrium applies at every timeframe — a daily swing has its own EQ, and within that, an H1 swing has its own. Understanding which zone you're in prevents the common mistake of buying high and selling low."
            },
        ]
    },
    {
        "name": "Propulsion Block",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Propulsion Block?",
                "a": "A Propulsion Block is a close cousin of the Order Block — it's a candle within a strong displacement move that has unusually high momentum. Unlike a standard OB (which is the last opposing candle before displacement), a Propulsion Block is a candle within the displacement itself that acts as a launching pad. When price retraces to a Propulsion Block, it tends to 'propel' price back in the displacement direction with force. Think of it as a fuel stop within an institutional move. Propulsion Blocks are especially useful when the standard Order Block is too far away for an efficient entry — the Propulsion Block gives you a closer entry point within the move itself."
            },
        ]
    },
    {
        "name": "Rejection Block",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Rejection Block in ICT?",
                "a": "A Rejection Block is identified by the wick (shadow) of a candle at a significant level — typically at a liquidity sweep or Order Block test. The wick portion represents prices that were rejected by the market, indicating institutional defense of that level. A long upper wick at a bearish OB shows aggressive selling; a long lower wick at a bullish OB shows aggressive buying. The Rejection Block zone is the wick itself — when price returns to this area, it often gets rejected again. Rejection Blocks are confirmation tools: they validate that smart money is active at a level. They're particularly useful for confirming Order Blocks and identifying failed breakouts."
            },
        ]
    },
    {
        "name": "SIBI and BISI",
        "alias": ["SIBI", "BISI"],
        "qa_pairs": [
            {
                "q": "What are SIBI and BISI in ICT?",
                "a": "SIBI (Sell-Side Imbalance, Buy-Side Inefficiency) is a bearish Fair Value Gap — it forms when aggressive selling creates an imbalance, leaving an inefficiency on the buy side. BISI (Buy-Side Imbalance, Sell-Side Inefficiency) is a bullish Fair Value Gap — aggressive buying creates an imbalance with an inefficiency on the sell side. These are ICT's technical terms for FVG direction. A BISI forms during bullish displacement and becomes a zone where price retests for longs. A SIBI forms during bearish displacement and becomes a zone for shorts. Understanding SIBI/BISI helps you quickly identify which side of the market created the imbalance and which direction to trade when price returns to fill it."
            },
        ]
    },
    {
        "name": "Immediate Rebalance",
        "alias": ["IRB"],
        "qa_pairs": [
            {
                "q": "What is an Immediate Rebalance?",
                "a": "An Immediate Rebalance (IRB) occurs when an FVG is filled on the very next candle after it forms — the third candle in the pattern immediately retraces into the gap before continuing. This is significant because it shows that the imbalance was so aggressively addressed that institutions didn't wait — they immediately moved to rebalance the price. An IRB provides an ultra-short-term entry: enter as the rebalance candle touches the FVG and target the continuation in the displacement direction. IRBs are most common during volatile Kill Zone periods and around news events. They require fast execution because the opportunity window is compressed to a single candle."
            },
        ]
    },
    {
        "name": "Dealing Range",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Dealing Range in ICT?",
                "a": "A Dealing Range in ICT is the price range between a significant swing high and swing low — it defines the current 'playing field' where price is being delivered. The dealing range has three key zones: Premium (above 50%), Equilibrium (the 50% midpoint), and Discount (below 50%). Understanding which dealing range you're in tells you where to buy and where to sell. A new dealing range is established after a Break of Structure or Market Structure Shift. Micro Dealing Ranges form within the larger range on lower timeframes. The current dealing range is always the most recent swing high to swing low that hasn't been broken — it's your context for determining premium vs discount for ANY entry."
            },
        ]
    },
    {
        "name": "Expansion Price Swing",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is an Expansion Price Swing?",
                "a": "An Expansion Price Swing is a strong, directional leg of price movement — the 'distribution' phase after accumulation and manipulation. It's the move where institutional intention is revealed and price travels from one liquidity pool to another. Expansion swings are characterized by displacement, large candle bodies, FVG creation, and clear directional commitment. They typically occur during Kill Zones after the Judas Swing (manipulation) has captured liquidity. From a measurement perspective, expansion swings are used for standard deviation projections: multiply the range of the expansion leg by 1.0, 1.5, 2.0, 2.5 SD to project targets. The expansion swing IS the trade — your goal is to catch it during distribution phase."
            },
        ]
    },
    {
        "name": "Standard Deviation Projections",
        "alias": ["SD"],
        "qa_pairs": [
            {
                "q": "How do Standard Deviation projections work in ICT?",
                "a": "Standard Deviation (SD) projections are ICT's method for projecting profit targets using mathematical multiples of a manipulation or expansion leg. Measure the range of the initial leg (manipulation move or first expansion swing). Then project multiples: -1.0 SD equals the same distance, -2.0 SD is double, -2.5 SD is 2.5x the distance. The -2.5 SD level is statistically the most significant — price often terminates here. Use the formula: Target = Reference Point - (Leg Range × SD Multiple) for bearish, or + for bullish. SD projections are applied to CBDR ranges, Asian Range expansions, and Judas Swing legs. They give you concrete profit targets instead of guessing where price will stop."
            },
        ]
    },
    {
        "name": "Separation Principle",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the Separation Principle?",
                "a": "The Separation Principle in ICT states that when price separates from a PD Array element (Order Block, FVG, etc.) with displacement, it confirms institutional commitment at that level. The 'separation' is the gap between the PD Array level and the current price created by the displacement candle. A clean separation means the displacement candle left behind an FVG between the PD Array and current price — this is the strongest confirmation. If price slowly drifts away from a level without separation (no FVG, no displacement), the setup is weak. The Separation Principle helps you filter high-probability from low-probability setups: no separation = no confidence in the trade."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  LIQUIDITY — Tier 1 gaps
    # ══════════════════════════════════════════════════════════
    {
        "name": "Buy-Side Liquidity",
        "alias": ["BSL"],
        "qa_pairs": [
            {
                "q": "What is Buy-Side Liquidity and where does it form?",
                "a": "Buy-Side Liquidity (BSL) is a cluster of resting buy stop orders above swing highs, equal highs, and obvious resistance levels. These orders come from two sources: short sellers' stop-losses (placed above recent highs) and breakout traders' buy entries (triggered on new highs). BSL forms at predictable levels — above any obvious swing high, above equal highs (double/triple tops), and above trendline resistance. Smart money targets these pools because they provide the volume needed to fill large sell orders. When institutions want to sell, they engineer price upward into BSL to trigger those buy stops, creating the counterparty volume they need. A BSL sweep followed by displacement downward is a classic sell signal."
            },
        ]
    },
    {
        "name": "Sell-Side Liquidity",
        "alias": ["SSL"],
        "qa_pairs": [
            {
                "q": "What is Sell-Side Liquidity and where does it form?",
                "a": "Sell-Side Liquidity (SSL) is a cluster of resting sell stop orders below swing lows, equal lows, and obvious support levels. These orders come from long traders' stop-losses (placed below recent lows) and breakout sellers' entries (triggered on new lows). SSL forms at predictable levels — below obvious swing lows, below equal lows (double/triple bottoms), and below trendline support. Institutions target SSL pools when they want to buy — they engineer price downward to trigger sell stops, creating the sell volume they need to fill their buy orders. An SSL sweep followed by bullish displacement is a classic buy signal. The sweeping of SSL is often the manipulation phase of Power of Three."
            },
        ]
    },
    {
        "name": "External Range Liquidity",
        "alias": ["ERL"],
        "qa_pairs": [
            {
                "q": "What is External Range Liquidity?",
                "a": "External Range Liquidity (ERL) refers to liquidity resting outside the current dealing range — above the range high (buy-side ERL) and below the range low (sell-side ERL). External liquidity is the 'target' of price delivery: when the market is trending, it's seeking external range liquidity. In a bullish market, price targets buy-side ERL above the current range. In a bearish market, it targets sell-side ERL below. The concept pairs with Internal Range Liquidity: price alternates between seeking internal and external liquidity. After sweeping external liquidity (making a new high or new low), price often reverses to address internal range inefficiencies. Understanding ERL vs IRL is key to determining whether price is reaching for new highs/lows or will retrace first."
            },
        ]
    },
    {
        "name": "Internal Range Liquidity",
        "alias": ["IRL"],
        "qa_pairs": [
            {
                "q": "What is Internal Range Liquidity?",
                "a": "Internal Range Liquidity (IRL) refers to the inefficiencies and PD arrays inside the current dealing range — unfilled FVGs, untested Order Blocks, and other imbalances between the range high and low. While External Range Liquidity sits at the extremes, IRL represents the 'unfinished business' within the range. After price sweeps external liquidity (makes a new high or low), it typically retraces to address IRL — filling FVGs, testing Order Blocks, and rebalancing imbalances. This IRL-to-ERL cycle is fundamental: price seeks external liquidity → reverses → fills internal inefficiencies → seeks external liquidity again. When you see price reverse from an ERL sweep, your targets should be the nearest IRL zones (unfilled FVGs, OBs)."
            },
        ]
    },
    {
        "name": "Draw on Liquidity",
        "alias": ["DOL"],
        "qa_pairs": [
            {
                "q": "What is Draw on Liquidity?",
                "a": "Draw on Liquidity (DOL) is the concept that price is always being 'drawn' toward a specific liquidity target — like a magnet. At any point in trade, you should identify the DOL: which liquidity pool is price heading toward? In a bullish scenario, the DOL is typically buy-side liquidity above (equal highs, swing highs, PDH, PWH). In a bearish scenario, the DOL is sell-side liquidity below. Identifying the DOL helps you determine your trade's target and confirms directional bias. If you can't identify a clear DOL, don't trade — the market needs a destination. The IPDA (Interbank Price Delivery Algorithm) uses 20/40/60-day lookback periods to determine which DOL is being targeted."
            },
        ]
    },
    {
        "name": "Liquidity Sweep",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Liquidity Sweep and how do I spot one?",
                "a": "A Liquidity Sweep is when price briefly trades beyond a key level (swing high/low, equal highs/lows) to trigger resting stop orders, then immediately reverses. You spot a sweep by looking for a wick that penetrates a level without a body close beyond it — price pokes through, grabs the stops, and snaps back. The sweep creates a 'wick' on the chart. Another telltale: volume spikes during the sweep as all those stops get triggered simultaneously. Sweeps are the mechanism behind the Judas Swing and manipulation phase. After a sweep, look for displacement in the opposite direction — that confirms smart money got what they needed (liquidity fill) and are now moving price in their intended direction."
            },
        ]
    },
    {
        "name": "Equal Highs and Equal Lows",
        "alias": ["EQH", "EQL"],
        "qa_pairs": [
            {
                "q": "Why are Equal Highs and Equal Lows so important in ICT?",
                "a": "Equal Highs (EQH) and Equal Lows (EQL) are the most concentrated liquidity pools on any chart. When price posts two or more highs at nearly the same level, retail traders see 'strong resistance' and place sell stops just below and breakout buys just above. This creates a dense cluster of orders. Same with Equal Lows — retail sees 'strong support' and stacks orders around them. Smart money sees these as targets, not barriers. Equal highs are buy-side liquidity targets; equal lows are sell-side liquidity targets. The more 'equal' the highs or lows (closer in price), the more concentrated the stops and the more attractive the target for institutions. When equal highs/lows get raided, expect a reversal — the liquidity has been captured."
            },
        ]
    },
    {
        "name": "Low Resistance Liquidity Run",
        "alias": ["LRLR"],
        "qa_pairs": [
            {
                "q": "What is a Low Resistance Liquidity Run?",
                "a": "A Low Resistance Liquidity Run (LRLR) is a directional move through price areas that offer minimal opposing order flow — essentially clear highway for price. These runs occur when there are few unfilled FVGs, no significant Order Blocks, and no major liquidity pools in the path. Price can move quickly through these zones because there's nothing to slow it down. Recognizing an LRLR helps you understand why price sometimes moves in a straight line without retracing — there are no obstacles. Conversely, when price approaches a zone with dense PD arrays (stacked OBs, unfilled FVGs), expect it to slow down, react, or reverse. LRLRs are common during expansion phases of the Power of Three and during strong trend days."
            },
        ]
    },
    {
        "name": "Stop Hunt",
        "alias": ["Equity Run"],
        "qa_pairs": [
            {
                "q": "What is a Stop Hunt in ICT methodology?",
                "a": "A Stop Hunt is an engineered price move designed specifically to trigger clustered stop-loss orders at predictable levels. Institutions know where retail traders place stops — below swing lows (for longs) and above swing highs (for shorts). A stop hunt drives price to those levels, triggers the stops (creating counterparty volume), then reverses. It's the mechanism behind the manipulation phase of Power of Three. Stop hunts typically occur during Kill Zones, especially around London Open and NY Open. The key distinction from a real breakout: a stop hunt reverses quickly after triggering stops and shows immediate displacement back. A real breakout shows continuation and follow-through after breaking the level."
            },
        ]
    },
    {
        "name": "High Resistance Liquidity Zone",
        "alias": ["HRLZ"],
        "qa_pairs": [
            {
                "q": "What is a High Resistance Liquidity Zone?",
                "a": "A High Resistance Liquidity Zone (HRLZ) is an area dense with PD array elements — stacked Order Blocks, unfilled FVGs, Breaker Blocks — that creates significant resistance to price movement. When price enters an HRLZ, it typically slows down, consolidates, or reverses because there are too many institutional reference points opposing the move. HRLZs are the opposite of Low Resistance Liquidity Runs. Identifying them helps you set realistic targets: don't expect price to blow through an HRLZ quickly. If your target is beyond an HRLZ, consider taking partial profits before it or wait for a clear break through the zone before adding to your position."
            },
        ]
    },
    {
        "name": "Pairing of Orders",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is Pairing of Orders in ICT?",
                "a": "Pairing of Orders is the concept that for every trade execution, there must be a matching counterparty — every buy needs a seller and every sell needs a buyer. This is why institutions engineer liquidity events: they need someone on the other side of their trade. When institutions want to buy, they need sellers — so they create bearish moves that trigger sell stops and induce retail selling. Those panicking retail sellers become the 'pair' for institutional buys. Understanding pairing explains WHY manipulation exists: it's not random, it's institutional order filling. Price always moves to areas where orders are paired — from liquidity pools to inefficiencies and back. The market is fundamentally an order-pairing engine."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  MARKET STRUCTURE — Tier 1 gaps
    # ══════════════════════════════════════════════════════════
    {
        "name": "Break of Structure",
        "alias": ["BOS"],
        "qa_pairs": [
            {
                "q": "What is a Break of Structure in ICT?",
                "a": "A Break of Structure (BOS) occurs when price breaks a previous swing point IN THE SAME DIRECTION as the prevailing trend with visible displacement. In a bullish trend (HH/HL), a BOS happens when price creates a new Higher High by breaking above the previous swing high. In a bearish trend (LH/LL), a BOS happens when price makes a new Lower Low. BOS is a continuation signal — it confirms the existing trend is still intact. The key requirement is displacement: the break must be decisive with strong candle bodies, not just a wick poke. After a BOS, look for the new Order Block and FVG created by the displacement for your entry on the retracement."
            },
            {
                "q": "How is BOS different from MSS and CHoCH?",
                "a": "BOS (Break of Structure) is a continuation signal — price breaks a swing point in the trending direction, confirming the trend continues. MSS (Market Structure Shift) is a reversal signal — price breaks a swing point AGAINST the current trend with displacement, signaling a potential direction change. CHoCH (Change of Character) is an early warning — it's a weaker break against the trend that alerts you to weakening momentum but isn't confirmed yet. The hierarchy: CHoCH is the first clue, MSS is the confirmation of reversal, BOS confirms the new trend. In practice: see CHoCH → get alert. See MSS with displacement → look for entries in the new direction. See BOS → confirms you're in a trend, add on retracements."
            },
        ]
    },
    {
        "name": "Change of Character",
        "alias": ["CHoCH", "CHOCH"],
        "qa_pairs": [
            {
                "q": "What is Change of Character in ICT?",
                "a": "Change of Character (CHoCH) is an early warning signal that the current trend may be weakening. It occurs when price violates a short-term swing point against the prevailing trend — but without strong displacement. For example, in a bullish trend making HH/HL, a CHoCH happens when price dips below a recent higher low without conviction. It's weaker than a full Market Structure Shift because it lacks displacement and may not indicate a true reversal. CHoCH tells you to be cautious — stop adding to the current trend direction, tighten stops, and watch for a full MSS confirmation. Many CHoCH signals resolve with the trend continuing, which is why you don't trade them alone."
            },
        ]
    },
    {
        "name": "Swing Failure Pattern",
        "alias": ["SFP"],
        "qa_pairs": [
            {
                "q": "What is a Swing Failure Pattern?",
                "a": "A Swing Failure Pattern (SFP) occurs when price trades beyond a previous swing point (sweeping the liquidity) but fails to close beyond it and reverses. It's essentially a failed breakout that captures liquidity. A bearish SFP: price wicks above a swing high (triggering buy stops) but the candle body closes back below the high. A bullish SFP: price wicks below a swing low (triggering sell stops) but closes back above. SFPs are powerful reversal signals because they combine a liquidity sweep with an immediate failure. They confirm that smart money used the breakout to fill orders in the opposite direction. SFPs at equal highs/lows or during Kill Zones are especially high-probability."
            },
        ]
    },
    {
        "name": "Swing High and Swing Low",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How do I properly identify Swing Highs and Swing Lows?",
                "a": "A Swing High is a candle with a high that is higher than the highs of the candles on both sides of it — it forms a peak in price action. A Swing Low is a candle with a low that is lower than the lows of the candles on both sides — it forms a trough. In ICT, swing points are critical because they define market structure (HH/HL/LH/LL pattern), they mark where liquidity rests (stops above swing highs, below swing lows), and they create the framework for your dealing range. When identifying swings, timeframe matters: H4 swing points carry more weight than M5 swings. The sequence of swing points tells you the trend: ascending swing highs and lows = bullish, descending = bearish."
            },
        ]
    },
    {
        "name": "Reversal Sequence",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Reversal Sequence in ICT?",
                "a": "A Reversal Sequence is the complete chain of events that marks a trend change. It follows a specific order: First, price sweeps a key liquidity level (e.g., SSL below an obvious low). Second, a Market Structure Shift occurs — price displaces through a recent swing point in the new direction, breaking the old trend's pattern. Third, price creates new PD arrays (FVGs, Order Blocks) during the displacement. Fourth, price retraces to these PD arrays for entry. Fifth, expansion follows in the new direction toward the new DOL. Understanding the full reversal sequence prevents premature entries — you wait for each step to confirm before committing capital. Skip a step and you're guessing."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  TIME / SESSIONS — Tier 1 gaps
    # ══════════════════════════════════════════════════════════
    {
        "name": "London Killzone",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What specifically happens during the London Killzone?",
                "a": "The London Killzone runs from 2:00 AM to 5:00 AM EST and is one of the two highest-probability trading windows. During London Open, the manipulation phase of Power of Three typically unfolds — the Judas Swing occurs here, sweeping the Asian session high or low to capture liquidity. Smart money enters positions during this window using the liquidity generated by the sweep. The London session creates the early directional move of the day, often establishing the daily high or low. Key setups during London: Judas Swing into OB/FVG entries, Asian Range sweep reversals, and Silver Bullet setups during the 3:00-4:00 AM EST window. London is where most of the day's manipulation happens."
            },
        ]
    },
    {
        "name": "New York AM Killzone",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What makes the NY AM Killzone special?",
                "a": "The New York AM Killzone runs from 7:00 or 8:30 AM to 11:00 AM EST and offers the highest volume and most reliable ICT setups. This is where the distribution phase of Power of Three plays out — the true directional move of the day. NY AM often confirms the direction established during London or engineers a counter-move if London was the manipulation. Key times within NY AM: 8:20 AM (CME Globex open), 9:30 AM (NYSE/NASDAQ open providing volatility injection), and 10:00-11:00 AM (the Silver Bullet window with its precision FVG entries). The 9:50-10:10 AM macro window is statistically one of the most reliable 20-minute intervals for institutional setup completion."
            },
        ]
    },
    {
        "name": "Macro Times",
        "alias": ["Algorithm Macros"],
        "qa_pairs": [
            {
                "q": "What are Macro Times in ICT?",
                "a": "Macro Times are specific 20-minute windows within Kill Zones where the IPDA algorithm is most active — seeking liquidity or repricing FVGs. The key macros are: 9:50-10:10 AM EST (the Silver Bullet macro within NY AM), 2:33-3:00 AM EST (the London macro), and 1:10-1:40 PM EST (the PM session macro). During these precise windows, PD array elements (Order Blocks, FVGs) are more likely to be respected, liquidity sweeps are more likely to occur, and setup completion has higher probability. Macro times add another layer of time confluence: a setup that forms during a Kill Zone AND during a macro window within that Kill Zone has significantly higher probability than one outside these precise intervals."
            },
        ]
    },
    {
        "name": "London Close Killzone",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What happens during the London Close Killzone?",
                "a": "The London Close Killzone runs from 10:00 AM to 12:00 PM EST and represents the transition from London to New York-only trading. During this window, London-based institutions close their positions, often causing reversals of the London move. If London drove price in one direction, London Close may see a partial or full retracement as those positions are wound down. This creates opportunities for counter-trend setups — but only for experienced traders who can read the transition correctly. The Dead Zone follows London Close (11:00-12:30 PM) where institutional activity drops significantly. Most ICT practitioners avoid trading during London Close unless they see very clear OB/FVG setups with strong confluence."
            },
        ]
    },
    {
        "name": "Tuesday High/Low of the Week",
        "alias": [],
        "qa_pairs": [
            {
                "q": "Why is Tuesday important for the weekly high or low?",
                "a": "In ICT methodology, Tuesday often marks the high or low of the trading week. This is because the weekly Power of Three typically completes its manipulation phase by Tuesday. Monday usually continues the previous week's action or consolidates (accumulation). Tuesday's manipulation sweep often captures weekly liquidity — either sweeping the previous week's high or low — establishing the week's extreme. Once Tuesday's liquidity event has occurred, Wednesday through Friday are the distribution phase where the true weekly move plays out. Practically, this means: if Tuesday makes a significant low after sweeping sell-side liquidity, expect a bullish weekly move from Wednesday onward. If Tuesday sweeps a high, expect bearish distribution."
            },
        ]
    },
    {
        "name": "Dead Zone",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the Dead Zone and why should I avoid it?",
                "a": "The Dead Zone is the period from 11:00 AM to 12:30 PM EST — the New York lunch break. During this window, institutional traders step away, volume drops dramatically, and price action becomes choppy, directionless noise. Trading during the Dead Zone is dangerous because Order Blocks and FVGs are less likely to be respected, spreads may widen, and false signals proliferate. The Dead Zone separates the NY AM Kill Zone from the NY PM Kill Zone. Smart money uses this period to reassess and reposition for the afternoon session. Wait for the NY PM Kill Zone (1:30 PM onward) to resume trading. Many losing trades come from boredom-driven entries during the Dead Zone."
            },
        ]
    },
    {
        "name": "IPDA True Day",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the IPDA True Day?",
                "a": "The IPDA True Day runs from midnight EST (00:00) to 3:00 PM EST, representing the full window of institutional price delivery. Unlike the retail trading day that follows exchange hours, the IPDA True Day captures the complete algorithm cycle: Asian session accumulation, London manipulation, and NY distribution all within one continuous 15-hour window. The midnight open is significant as a reference level — it's the true day's opening price, not 9:30 AM. The IPDA True Day ends at 3:00 PM EST because by then, the day's institutional objectives (liquidity targets, FVG fills, delivery to PD arrays) should be complete. After 3:00 PM is the CBDR period for the next day's setup."
            },
        ]
    },
    {
        "name": "Sunday Opening Price",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How is the Sunday Opening Price used in ICT?",
                "a": "The Sunday Opening Price is the first price printed when markets open Sunday evening (5:00 PM EST in forex). It serves as a weekly reference level — a bias filter for the entire week. If price is above the Sunday open during the week, bullish bias is supported. If below, bearish bias is supported. The Sunday open also helps identify the New Week Opening Gap (NWOG) — the difference between Friday's close and Sunday's open. When combined with Tuesday's high/low significance, the Sunday open gives you context: has the market gapped above Friday's close (bullish sentiment shift)? Has it gapped below (bearish)? The gap direction often aligns with the early week manipulation before the true weekly move begins."
            },
        ]
    },
    {
        "name": "Ninety Minute Cycles",
        "alias": ["90-Minute Cycles"],
        "qa_pairs": [
            {
                "q": "What are the 90-minute cycles in ICT?",
                "a": "ICT identifies 90-minute cycles as natural rhythms in the IPDA algorithm. Price tends to complete a full cycle — from accumulation through manipulation to distribution — within approximately 90 minutes. This means within each Kill Zone, you can often observe a mini Power of Three playing out over 90 minutes. The cycles help with timing: if a manipulation phase (Judas Swing) occurs in the first 30 minutes of a Kill Zone, the distribution move typically completes within the next 60 minutes. Knowing this prevents holding too long — if your trade hasn't moved in your direction within 90 minutes of entry, the cycle may have completed and a new one is beginning. The cycles also help structure your analysis windows."
            },
        ]
    },
    {
        "name": "Quarterly Shifts",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What are Quarterly Shifts in ICT?",
                "a": "Quarterly Shifts refer to the tendency for markets to shift directional bias at the beginning of each calendar quarter (January, April, July, October). Institutional portfolio rebalancing occurs at quarter boundaries, where funds reallocate capital across asset classes. This creates significant shifts in order flow that can reverse multi-month trends. March-April transitions are particularly significant as they coincide with fiscal year-end for many institutions. These shifts often manifest as major swing points on weekly/monthly charts. For practical trading, be especially cautious at quarter boundaries: trends that seemed unstoppable may reverse as institutional flows change. Combine with seasonal tendencies for higher-timeframe directional bias."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  TRADING MODELS — Tier 1 gaps
    # ══════════════════════════════════════════════════════════
    {
        "name": "Accumulation",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the Accumulation phase in ICT?",
                "a": "Accumulation is the first phase of the Power of Three (AMD). During accumulation, smart money quietly builds positions without significantly moving price — creating a tight consolidation range. This typically occurs during the Asian session (8 PM - midnight EST) or during the CBDR period (2-8 PM EST). The range created during accumulation sets up the playing field for the next two phases. A tight accumulation range (small CBDR, narrow Asian range) signals that a large expansion move is coming — smart money is coiled and ready. Accumulation looks boring on a chart: small candles, no direction, range-bound price. But it's the most important phase because it tells you something big is being prepared. Identify accumulation → expect manipulation → position for distribution."
            },
        ]
    },
    {
        "name": "Distribution",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the Distribution phase in ICT?",
                "a": "Distribution is the third and final phase of the Power of Three (AMD) — the 'real move' where smart money achieves its objective. After accumulation (quiet building) and manipulation (false move/Judas Swing), distribution is the expansion where institutional order flow drives price aggressively toward the target liquidity pool. Distribution typically occurs during the London session or NY AM Kill Zone. Characteristics: strong displacement, FVG creation, clear directional candles, and price traveling from one liquidity target to another. Your goal as a trader is to enter during late manipulation or early distribution — catching the expansion move while avoiding the trap. Distribution continues until price reaches its draw on liquidity target or encounters a high-resistance zone."
            },
        ]
    },
    {
        "name": "Turtle Soup",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is Turtle Soup in ICT?",
                "a": "Turtle Soup is a liquidity sweep reversal pattern named after the Turtle Traders' breakout system. The original Turtles bought breakouts above recent highs — ICT's Turtle Soup exploits those breakout entries as liquidity targets. The pattern: price breaks above a recent high (or below a recent low), triggering breakout entries and stop-losses. But instead of following through, price immediately reverses. The breakout traders are trapped, and their stops become fuel for the reversal move. Turtle Soup Plus One is the variant where the reversal happens the next trading day rather than immediately. Turtle Soup setups are most reliable when they occur during Kill Zones after sweeping obvious liquidity levels. The pattern validates that the breakout was a manipulation, not an expansion."
            },
        ]
    },
    {
        "name": "Unicorn Setup",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Unicorn Setup in ICT?",
                "a": "A Unicorn Setup is a high-confluence entry where a Breaker Block and a Fair Value Gap converge at the same price level — creating a powerful dual-confirmation zone. It's called 'Unicorn' because the confluence is rare and highly effective. The setup forms when: a failed Order Block creates a Breaker Block, and the displacement that created the Breaker also leaves behind an FVG overlapping the same zone. When price retraces to this area, you have two institutional reasons for a reaction: the Breaker provides structural support/resistance and the FVG provides algorithmic draw. Enter at the overlapping zone with a tight stop beyond both elements. Unicorn setups during Kill Zones are considered among the highest-probability entries in ICT methodology."
            },
        ]
    },
    {
        "name": "A Plus Setup",
        "alias": ["A+ Setup", "A+ Template"],
        "qa_pairs": [
            {
                "q": "What makes an A+ Setup in ICT?",
                "a": "An A+ Setup (A+ Template Trade) requires seven points of confluence — it's the highest-probability trade in ICT methodology. The seven confluences are: 1) Higher Timeframe bias alignment (Daily/H4 direction), 2) Kill Zone timing (London or NY AM), 3) Liquidity sweep (BSL or SSL captured), 4) Market Structure Shift with displacement, 5) PD Array entry (Order Block or FVG in discount/premium), 6) SMT Divergence confirmation between correlated pairs, 7) Macro time alignment. All seven must be present simultaneously. With all confluences met, the risk:reward is typically 3:1 or better with a high win rate. A+ setups don't appear every day — sometimes you wait all week for one. Patience for A+ setups is what separates profitable ICT traders from struggling ones."
            },
        ]
    },
    {
        "name": "Model 7 - Universal Trading Model",
        "alias": ["ICT Model 7"],
        "qa_pairs": [
            {
                "q": "What is ICT Model 7, the Universal Trading Model?",
                "a": "ICT Model 7 is the Universal Trading Model — a flexible framework that works across all asset classes and timeframes. It centers on the trade plan and algorithmic theory: price is delivered by an algorithm seeking liquidity and balancing inefficiencies. The model sequence: 1) Identify the higher timeframe bias using market structure. 2) Determine the Draw on Liquidity (where is price headed?). 3) Wait for a Kill Zone. 4) Identify the manipulation phase (Judas Swing or liquidity sweep). 5) Confirm with MSS + displacement on the lower timeframe. 6) Enter on the OB or FVG created by the displacement. 7) Target the DOL with stops below the manipulation extreme. Model 7 is called 'universal' because it encapsulates the core ICT decision process applicable to forex, indices, commodities — any market."
            },
        ]
    },
    {
        "name": "Model 9 - One Shot One Kill",
        "alias": ["OSOK", "ICT Model 9"],
        "qa_pairs": [
            {
                "q": "What is ICT Model 9, One Shot One Kill?",
                "a": "ICT Model 9, One Shot One Kill (OSOK), is designed to capture 50-75 pips per week with a single high-conviction trade. The philosophy: quantity of trades doesn't matter, quality does. Model 9 requires waiting for the week's highest-probability setup and executing with precision. The sequence: 1) Weekly analysis to determine the week's likely bias and target (using Sunday open, weekly market structure, HTF PD arrays). 2) Wait for Tuesday's high/low establishment. 3) Identify the optimal entry using a confluence-rich setup during a Kill Zone. 4) Execute one trade targeting 50-75 pips. 5) Once achieved, you're done for the week. The discipline of Model 9 prevents overtrading and forces you to be selective. The target of 50-75 pips per week compounds to exceptional annual returns."
            },
        ]
    },
    {
        "name": "Model 12 - Scalping",
        "alias": ["ICT Model 12"],
        "qa_pairs": [
            {
                "q": "What is ICT Model 12, the Scalping Model?",
                "a": "ICT Model 12 is the Scalping/Intraday Model targeting 20 pips per trade using Order Block + FVG confluence entries. The model operates on 5-minute and 1-minute charts during Kill Zones. The sequence: 1) Confirm HTF bias on H1/H4. 2) During a Kill Zone, identify a liquidity sweep on M15. 3) Drop to M5 — look for MSS with displacement. 4) Mark the M5 Order Block and FVG created by displacement. 5) Enter when price retraces to the OB+FVG confluence on M1/M5. 6) Stop below the OB, target 20 pips. The small target means tight risk management — typically 5-10 pip stops for 2:1 to 4:1 R:R. Model 12 is designed for quick, precise executions during the most active windows. Multiple setups per day are possible but never take more than 3."
            },
        ]
    },
    {
        "name": "CBDR and Asia Range SD Method",
        "alias": ["CBDR+Asia SD"],
        "qa_pairs": [
            {
                "q": "How does the CBDR and Asia Range Standard Deviation method work?",
                "a": "The CBDR+Asia SD method projects the next session's expansion targets using standard deviation multiples of the CBDR and Asian Range. Step 1: Measure the CBDR (2-8 PM EST range). Step 2: Measure the Asian Range (8 PM - midnight EST). Step 3: If both are tight (CBDR under 40 pips, Asian Range under 30 pips), expect strong expansion. Step 4: Apply SD projections from the Asian Range high/low: ±1.0 SD, ±2.0 SD, ±2.5 SD, ±4.0 SD. The -2.5 SD level is the statistically most significant target — the 'terminus' where price often exhausts. Direction is determined by higher timeframe bias. The tighter the CBDR+Asia combination, the more reliable the SD projection targets. This method gives you concrete profit-taking levels before the trading day even begins."
            },
        ]
    },
    {
        "name": "HTF Sweep to LTF Entry",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does the HTF Sweep to LTF Entry process work?",
                "a": "The HTF Sweep to LTF Entry is a core ICT workflow for high-probability trades. Step 1: On the higher timeframe (H4/Daily), identify a key liquidity level being targeted — BSL above swing highs or SSL below swing lows. Step 2: Wait for price to sweep that HTF liquidity level (the wick pokes beyond it). Step 3: Drop to the lower timeframe (M15/M5) and look for a Market Structure Shift confirming the reversal. Step 4: On the LTF, identify the Order Block or FVG created by the MSS displacement. Step 5: Enter on the LTF PD array with your stop beyond the HTF sweep extreme. Step 6: Target the next HTF liquidity pool in the opposite direction. This process combines HTF context (where) with LTF precision (when) for optimal entries."
            },
        ]
    },
    {
        "name": "ICT 2022 Model",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the ICT 2022 Mentorship Model?",
                "a": "The ICT 2022 Model is the refined framework from ICT's 2022 mentorship series. It simplifies earlier concepts into a focused workflow: 1) Time over Price — Kill Zones and macro times are the primary filter. 2) Liquidity and Efficiency — price always moves between external liquidity targets (swing highs/lows) and internal efficiency targets (FVGs). 3) The three core mechanics are: a) External Range Liquidity draws price to swing extremes, b) Internal Range Liquidity (unfilled FVGs) draws price back for rebalancing, c) Price alternates between external and internal targets in a predictable cycle. The 2022 model emphasizes FVGs over Order Blocks as the primary entry mechanism, and uses displacement as the universal confirmation signal."
            },
        ]
    },
    {
        "name": "London Open Manipulation Model",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does the London Open Manipulation Model work?",
                "a": "The London Open Manipulation Model describes the typical price behavior between 2:00-5:00 AM EST. The sequence: 1) During the Asian session (accumulation), smart money established positions and created a tight range. 2) At London Open, the IPDA algorithm engineers a Judas Swing — a false move that sweeps the Asian high or low to capture resting liquidity. 3) After capturing liquidity, price reverses with displacement (creating FVGs and Order Blocks). 4) The true London move begins — expansion toward the day's primary target. To trade this model: identify the Asian Range, determine HTF bias, wait for the Judas Swing to sweep the wrong side, then enter on the reversal MSS with targets at opposing liquidity. The London manipulation is the most reliable daily Judas Swing."
            },
        ]
    },
    {
        "name": "ICT Reversal Model",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the ICT Reversal Model?",
                "a": "The ICT Reversal Model describes the conditions for a high-probability reversal trade. Requirements: 1) Price reaches a significant HTF PD array (Order Block, FVG on Daily/H4). 2) Liquidity sweep occurs — price wicks beyond a key level, capturing stops. 3) Displacement in the opposite direction follows immediately. 4) MSS confirms the reversal on LTF (M15/M5). 5) New PD arrays (OBs, FVGs) are created during the reversal displacement. 6) Enter on retracement to the new PD arrays. 7) Target the next significant liquidity pool in the reversal direction. Reversals are higher risk than continuation trades, so they require more confluence — ideally Kill Zone timing, SMT Divergence confirmation, and A+ template criteria. Never anticipate a reversal without seeing all five confirmation signals."
            },
        ]
    },
    {
        "name": "ICT Continuation Model",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the ICT Continuation Model?",
                "a": "The ICT Continuation Model is used when the trend is established and you're looking to add to positions or enter in the trend direction on pullbacks. The sequence: 1) Confirm HTF trend (HH/HL for bullish, LH/LL for bearish). 2) Wait for price to retrace to a PD array in the trend direction — a discount FVG for longs, premium FVG for shorts. 3) Confirm BOS (Break of Structure) in the trend direction on LTF. 4) Enter at the OB or FVG created by the continuation displacement. 5) Stop below the retracement low (bullish) or above the retracement high (bearish). 6) Target the next external range liquidity. Continuation setups are generally higher probability than reversals because you're trading with institutional order flow, not against it."
            },
        ]
    },
    {
        "name": "Day Trading Model",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the ICT Day Trading Model?",
                "a": "The ICT Day Trading Model structures the entire trading day into actionable phases. Pre-session (night before): determine daily bias from HTF analysis, measure CBDR, identify key liquidity targets. Asian session: monitor accumulation, note the range high/low. London Open (2-5 AM EST): watch for Judas Swing manipulation, identify entry setups. NY AM (8:30-11 AM EST): execute primary trades during the distribution phase, target 20-75 pips depending on your model (12/11/9). Dead Zone (11-12:30 PM): no trading, review. NY PM (1:30-4 PM EST): secondary setups only, lighter position sizes. Post-session: journal, analyze, prepare for tomorrow. The Day Trading Model is the master schedule that all other ICT models operate within."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  INSTITUTIONAL / ALGORITHMIC — Tier 1 gaps
    # ══════════════════════════════════════════════════════════
    {
        "name": "IPDA",
        "alias": ["Interbank Price Delivery Algorithm"],
        "qa_pairs": [
            {
                "q": "What is the IPDA?",
                "a": "The IPDA (Interbank Price Delivery Algorithm) is ICT's foundational concept — the idea that price movements are not random but algorithmically driven by institutional code. The IPDA has two primary directives: seek liquidity (resting orders at predictable levels) and rebalance inefficiencies (fill FVGs and tested PD arrays). It operates on three data ranges: Short-term (20-day lookback for recent swing reference), Intermediate (40-day lookback for context), and Long-term (60-day lookback for major levels). The algorithm delivers price from one liquidity pool to another, filling FVGs along the way. Every concept in ICT — Kill Zones, PD arrays, market structure — is a lens for reading the IPDA's intentions. Understanding the IPDA means understanding that price is engineered, not random."
            },
        ]
    },
    {
        "name": "Smart Money Reversal",
        "alias": ["SMR"],
        "qa_pairs": [
            {
                "q": "What is a Smart Money Reversal?",
                "a": "A Smart Money Reversal (SMR) is the moment in the Market Maker Buy or Sell Model where institutional order flow changes direction. In the MMSM, the SMR occurs after price sweeps buy-side liquidity — institutions have filled their sell orders using breakout buyers' liquidity, and now they reverse price downward. In the MMBM, the SMR happens after price sweeps sell-side liquidity. The SMR is confirmed by: a liquidity sweep at the model's extreme, displacement in the new direction, and a Market Structure Shift on multiple timeframes. The SMR is the turning point you're waiting for — it's when smart money reveals their true intention. Enter on the retracement after the SMR with targets at the model's terminus (opposing liquidity)."
            },
        ]
    },
    {
        "name": "Daily Bias",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How do I determine Daily Bias in ICT?",
                "a": "Daily Bias is your directional conviction for the trading day — bullish or bearish. Determine it the night before using: 1) Weekly/Daily market structure — is price making HH/HL (bullish) or LH/LL (bearish)? 2) Where is the nearest Draw on Liquidity? BSL above suggests bullish bias; SSL below suggests bearish. 3) Measure the CBDR — a tight CBDR means expansion is coming, but you still need direction from points 1 and 2. 4) Check unfilled HTF FVGs and untested Order Blocks — price tends to target these. 5) Review the economic calendar for catalysts. 6) Use intermarket correlation — DXY inverse for EUR/USD, ES/NQ correlation for indices. Once bias is set, only take trades in that direction. No bias = no trading. Bias is the first filter before any entry."
            },
        ]
    },
    {
        "name": "Confluence and Confluence Scoring",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does confluence scoring work in ICT?",
                "a": "Confluence scoring is a systematic way to evaluate trade quality before entering. Each ICT factor adds points to your score: HTF bias alignment (+15), Kill Zone timing (+15), Liquidity sweep (+15), MSS with displacement (+10), PD Array entry (OB/FVG) (+10), Premium/Discount zone alignment (+10), SMT Divergence (+15), Macro time alignment (+10). A minimum score of 50 is required for entry; 70+ is an A+ setup. The scoring system prevents emotional trading — you can't enter if the math doesn't support it. Pre-trade scoring should be written in your journal BEFORE entry, not after. If you can't articulate which confluences are present and calculate the score, you don't understand the setup well enough to trade it."
            },
        ]
    },
    {
        "name": "Reaccumulation and Redistribution",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What are Reaccumulation and Redistribution in ICT?",
                "a": "Reaccumulation is a pause within a bullish trend where smart money adds to their long positions before the next expansion leg up. It looks like a consolidation or minor pullback that retests PD arrays (OBs, FVGs) before continuing the uptrend. Redistribution is the bearish counterpart — a pause within a downtrend where smart money adds to shorts before the next leg down. Both are distinct from accumulation/distribution at trend extremes: reaccumulation happens mid-trend (bullish), redistribution happens mid-trend (bearish). Identifying these correctly prevents you from taking counter-trend trades during what is actually a healthy trend continuation. Key tell: during reaccumulation/redistribution, internal range liquidity gets addressed (FVGs filled, OBs tested) before the trend resumes."
            },
        ]
    },
    {
        "name": "Intermarket Analysis",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does Intermarket Analysis work in ICT?",
                "a": "Intermarket Analysis in ICT involves correlating multiple instruments to confirm directional bias and spot SMT Divergence. Key correlations: EUR/USD and GBP/USD are positively correlated — they should move together. If one makes a new low and the other doesn't, that's SMT Divergence signaling a reversal. DXY (Dollar Index) is inversely correlated with EUR/USD — dollar strength means euro weakness. ES (S&P) and NQ (Nasdaq) should move together — divergence between them signals institutional activity. Bond yields affect equity and forex markets. Use intermarket analysis for bias confirmation — if EUR/USD is bearish and DXY is bullish, your bias has multi-instrument support. Conflicting signals across markets mean sit on your hands and wait for clarity."
            },
        ]
    },
    {
        "name": "Economic Calendar as Smokescreen",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does ICT view economic news releases?",
                "a": "In ICT methodology, economic news events (NFP, FOMC, CPI) are viewed as smokescreens — pretexts that institutions use to execute pre-planned liquidity events. The narrative is that smart money already knows the outcome (or has positioned for multiple scenarios) and uses the volatility injection from news to capture liquidity and execute their plan. Price was already headed toward its target; news just provides the catalyst and volume to get there faster. Practically: avoid entering during high-impact news (30 minutes before/after). But use the post-news displacement as confirmation. If news creates a sweep of liquidity and then displacement in the opposite direction, that aligns with the Judas Swing concept. The move after news settles is often the tradeable move."
            },
        ]
    },
    {
        "name": "Weekly Profiles and Market Maker Templating",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What are Weekly Profiles in ICT?",
                "a": "ICT identifies recurring weekly patterns called Weekly Profiles. The two primary profiles: Type A Classic Buy Day — price opens the week near the low, consolidates Monday, establishes the low by Tuesday (sweeping SSL), then rallies Wednesday through Friday toward buy-side liquidity. Type B Classic Sell Day — price opens near the high, establishes the week's high by Tuesday (sweeping BSL), then sells off toward sell-side liquidity. Market Maker Templating overlay these profiles onto the weekly chart to predict which days will be distribution days vs manipulation days. By studying which profile the current week matches (based on Sunday open position, Monday's behavior), you can anticipate Tuesday's manipulation and position for Wednesday-Friday distribution."
            },
        ]
    },
    {
        "name": "Change in State of Delivery",
        "alias": ["CISD"],
        "qa_pairs": [
            {
                "q": "What is Change in State of Delivery?",
                "a": "Change in State of Delivery (CISD) refers to the transition point where price switches from seeking one type of liquidity to another — or from trending to rebalancing. A CISD occurs when price completes a liquidity run (sweeps external range liquidity) and shifts to rebalancing mode (addressing internal range FVGs). Or when price finishes filling an FVG and begins seeking the next liquidity pool. CISD is essentially the 'gear shift' in the IPDA algorithm. Recognizing CISD helps you understand when your trade's target has been reached and a new price delivery cycle begins. After a CISD, the playbook changes — your old PD arrays may become irrelevant as new ones form in the new phase."
            },
        ]
    },
    {
        "name": "Data Ranges and Lookback",
        "alias": ["20/40/60 Day Lookback"],
        "qa_pairs": [
            {
                "q": "What are the IPDA Data Ranges?",
                "a": "The IPDA uses three lookback periods to determine which levels are relevant for current price delivery: the 20-Day Lookback (short-term) captures recent swing highs/lows and FVGs — these are the most immediately relevant levels. The 40-Day Lookback (intermediate) provides broader context including multi-week swing points and untested PD arrays. The 60-Day Lookback (long-term) identifies major institutional levels, quarterly reference points, and high-timeframe liquidity targets. To use these: on your Daily chart, look back 20/40/60 days and identify the key highs, lows, and unfilled FVGs at each range. The IPDA will target these levels in order of proximity and importance. The 60-day lookback range is often called the UPOD (Uniform Price Objective Data) range."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  RISK MANAGEMENT — all gaps (0/12 trained)
    # ══════════════════════════════════════════════════════════
    {
        "name": "Position Sizing",
        "alias": ["Position Size Formula"],
        "qa_pairs": [
            {
                "q": "How should I calculate position size in ICT?",
                "a": "Position sizing in ICT follows a strict formula: Risk Amount = Account Balance × Risk Percentage. Position Size = Risk Amount ÷ Stop Distance (in pips) ÷ Pip Value. The maximum risk per trade is 1-2% of your account — never more. With ICT's tight stop placements (below OBs, beyond FVGs), you can get precise stop distances, which allows for larger position sizes while maintaining the same dollar risk. For example: $10,000 account, 1% risk = $100 per trade. If your stop is 10 pips and pip value is $10/pip, you trade 1 lot. If your stop is 20 pips, trade 0.5 lots. The stop distance is determined by your entry point — OB entries typically have tighter stops than FVG entries."
            },
        ]
    },
    {
        "name": "Two Strike Rule",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is the Two Strike Rule?",
                "a": "The Two Strike Rule is a risk management discipline: if you take two consecutive losing trades in a single day, you stop trading for the rest of that day. No exceptions. The logic: two losses in a row means either your bias is wrong, you're not reading the market correctly, or conditions are unfavorable. Continuing to trade in this state leads to revenge trading and compounding losses. After two strikes, close your charts, review your journal, and prepare for the next day. Combined with the maximum 3 trades per day rule, the Two Strike Rule protects your capital and psychology. Most of a trader's biggest losing days happen when they refuse to stop after early losses. The rule prevents catastrophic drawdowns."
            },
        ]
    },
    {
        "name": "Maximum 3 Trades Per Day",
        "alias": [],
        "qa_pairs": [
            {
                "q": "Why does ICT limit you to 3 trades per day maximum?",
                "a": "The 3-trade daily maximum prevents overtrading — the number one account killer. Here's the logic: there are only 2-3 legitimate Kill Zone opportunities per day (London Open, NY AM, sometimes NY PM). With only 2-3 high-probability windows, taking more than 3 trades means you're forcing setups outside optimal times. The rule pairs with the Two Strike Rule: if you lose your first two, you're done. If you win your first and lose the second, you have one more shot. If you hit your daily target on the first trade, consider stopping. Quality over quantity is fundamental to ICT profitability. Many successful ICT traders take only 1-2 trades per day, with some (Model 9 practitioners) taking only 1-2 per WEEK."
            },
        ]
    },
    {
        "name": "Pre-Trade Scoring",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does Pre-Trade Scoring work?",
                "a": "Pre-Trade Scoring is a 10-point checklist completed BEFORE entering any trade. Points are awarded for: HTF Bias alignment (is the trade with the main trend?), Kill Zone timing (is it during an optimal window?), Liquidity sweep confirmation (has a level been captured?), MSS/Displacement (has structure shifted?), PD Array entry (are you entering at an OB/FVG?), Premium/Discount alignment (buying in discount or selling in premium?), SMT Divergence (do correlated pairs confirm?), CBDR context (is the range tight signaling expansion?), Macro time alignment (within a 20-min macro window?), and Risk:Reward ratio (minimum 2:1?). Score below 6: do not enter. Score 6-7: acceptable with caution. Score 8-10: high conviction. Write the score in your journal before every trade."
            },
        ]
    },
    {
        "name": "Stop Placement Rules",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What are the ICT stop placement rules?",
                "a": "ICT stop placement prioritizes precision over arbitrary distances. For Order Block entries: place your stop just beyond the OB's extreme — below the OB body low (bullish) or above the OB body high (bearish). NOT beyond the entire OB wick — the body is sufficient. For FVG entries: stop just beyond the FVG boundary, not through the entire gap plus buffer. For OTE entries: stop just below the 100% retracement level. Key principle: stops should be placed at levels where, if hit, would mean your trade thesis is invalidated. If the OB fails, the trade thesis is wrong. Adding unnecessary buffer to your stop reduces your risk:reward ratio and means you're not confident in your level. Let the level tell you where the stop belongs."
            },
        ]
    },
    {
        "name": "ADR Targeting",
        "alias": ["Average Daily Range"],
        "qa_pairs": [
            {
                "q": "How do I use ADR for profit targeting?",
                "a": "The Average Daily Range (ADR) tells you how many pips a pair typically moves in a day. Target 65-70% of the ADR for your daily profit objective — this is ICT's sweet spot. Calculate: take the last 5-day ADR (average of each day's high-to-low range). If EUR/USD has a 90-pip ADR, your daily target is 58-63 pips. Don't try to capture 100% of the ADR — that requires catching the exact high and low, which is unrealistic. The 65-70% target gives you room for entry and exit imprecision while still capturing the meat of the move. Once you've achieved 65% of ADR, consider closing or trailing tightly. Going for more becomes diminishing returns and risks giving back profits."
            },
        ]
    },
    {
        "name": "Partial Exits and Profit Taking",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How should I take partial profits in ICT?",
                "a": "ICT profit-taking follows the 80/20 rule: take 80% of your position at a reasonable target, let 20% run for a potential home run. The process: 1) Enter full position at your PD array entry (OB/FVG). 2) When price reaches the first significant level (1.0 SD, or 50% of the way to your final target), take 50-80% off. 3) Move your stop to breakeven on the remaining position. 4) Let the runner target the final DOL (2.0-2.5 SD, or the opposing liquidity pool). This protects your profits while giving you exposure to the full move. The psychological benefit is huge: knowing you've already banked profit lets you hold the runner calmly. Never move your stop to breakeven before taking partials — that's the quickest way to get stopped out flat."
            },
        ]
    },
    {
        "name": "Equity Curve Leveling",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is Equity Curve Leveling?",
                "a": "Equity Curve Leveling is the practice of maintaining consistent risk per trade regardless of whether you're on a winning or losing streak. Many traders increase position sizes after wins (overconfidence) or after losses (revenge trading) — both destroy accounts. ICT advocates keeping your risk percentage fixed (1-2% per trade) and letting your absolute dollar risk grow naturally as your account grows. If you're on a losing streak, the Two Strike Rule stops you before significant damage. If you're winning, your account grows and 1% of a larger account equals larger positions automatically. The goal: a smooth, upward-sloping equity curve without the spikes and crashes that come from emotional position sizing."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  ANTI-PATTERNS — all gaps (0/8 trained)
    # ══════════════════════════════════════════════════════════
    {
        "name": "False Order Block",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How do I identify a False Order Block?",
                "a": "A False Order Block lacks the key ingredient that makes a real OB: displacement. A candle that looks like an OB but isn't followed by strong, impulsive price movement away from it is a False OB. Other signs: no FVG was created after the supposed OB, the move away was gradual rather than explosive, or the 'OB' formed outside a Kill Zone. False OBs are common during choppy, range-bound conditions (like the Dead Zone). If price returns to a supposed OB and slices right through it without even a wick rejection, it was never a real OB — institutions didn't place orders there. Before trading any Order Block, verify: Was there displacement? Did it create an FVG? Did it form during a Kill Zone? If not, it's likely false."
            },
        ]
    },
    {
        "name": "Fighting the Higher Timeframe",
        "alias": ["Fighting HTF"],
        "qa_pairs": [
            {
                "q": "Why is Fighting the Higher Timeframe a critical mistake?",
                "a": "Fighting the HTF means taking trades against the higher timeframe directional bias — shorting when the Daily is bullish, or buying when the H4 is bearish. This is one of the most destructive mistakes in ICT trading. The higher timeframe controls the direction; the lower timeframe provides entries. When you trade against the HTF, you're trading against institutional order flow — the very force that makes ICT setups work. Even if you see a clean LTF setup in the wrong direction, it has significantly lower probability and worse risk:reward. The solution: always start your analysis from the HTF. Determine Daily/H4 bias FIRST. Then only look for LTF setups that align with that bias. If HTF and LTF disagree, don't trade — wait for alignment."
            },
        ]
    },
    {
        "name": "Overtrading",
        "alias": [],
        "qa_pairs": [
            {
                "q": "How does ICT methodology prevent overtrading?",
                "a": "ICT builds multiple guardrails against overtrading: First, Kill Zone timing limits — you can only trade during 4-5 specific windows, not all day. Second, the 3-trade daily maximum caps your activity. Third, the Two Strike Rule stops you after two consecutive losses. Fourth, the pre-trade scoring system prevents entering weak setups. Fifth, the A+ template requires seven confluences — most market conditions don't meet this bar. Sixth, Model 9 (One Shot One Kill) philosophically limits you to 1-2 trades per WEEK. Overtrading is usually driven by boredom, FOMO, or revenge. ICT's structure makes overtrading physically difficult to do if you follow the rules. The traders who struggle with overtrading haven't fully committed to the framework."
            },
        ]
    },
    {
        "name": "Chased Move and Asia Breakout Trap",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What is a Chased Move and the Asia Breakout Trap?",
                "a": "A Chased Move is when you enter a trade AFTER displacement has already occurred — buying after a big rally or selling after a big drop. By chasing, you're paying premium prices for entries that should have been taken at PD arrays. The proper ICT approach: wait for the retracement to an OB or FVG after displacement, never chase the displacement itself. The Asia Breakout Trap is a specific variant: retail traders see price break above or below the Asian Range and enter in the breakout direction. But this breakout is often the manipulation phase (Judas Swing) of Power of Three — designed to trap breakout traders before reversing. The fix: never trade the initial Asian Range breakout. Wait for the reversal displacement that reveals the true direction."
            },
        ]
    },
    {
        "name": "News Gamble and Revenge Trade",
        "alias": [],
        "qa_pairs": [
            {
                "q": "What are News Gambling and Revenge Trading?",
                "a": "News Gambling is entering trades right before high-impact news events (NFP, FOMC, CPI) hoping to catch the volatility. This violates ICT principles because: news spreads widen dramatically (eating your stop), price can spike both ways in seconds (triggering stops in both directions), and the initial move is often the manipulation — not the true direction. Wait 15-30 minutes after news for price to settle, then trade the post-news displacement. Revenge Trading is entering immediately after a losing trade trying to 'make it back.' It ignores all ICT rules — no Kill Zone timing, no confluence scoring, no patience. Revenge trades have the lowest win rate of any trade type because they're emotionally driven. The Two Strike Rule specifically prevents revenge trading."
            },
        ]
    },
    {
        "name": "Mitigated FVG One-Time Use",
        "alias": [],
        "qa_pairs": [
            {
                "q": "Why is a mitigated FVG considered one-time use only?",
                "a": "In ICT methodology, once an FVG has been filled (price has returned and traded through the gap), it is considered mitigated and should NOT be used for a second entry. The logic: the imbalance that created the FVG has been rebalanced — both buyers and sellers have now transacted at those prices. There's no remaining inefficiency to draw price back again. Trading a mitigated FVG is like trying to fill a cup that's already full. However, if the fill created a new displacement that left behind a NEW FVG overlapping the old one, that new FVG is valid. Also, partial mitigation (price entered the FVG but didn't fully close through it) means the FVG is still active. Only full body-close through the entire FVG constitutes full mitigation."
            },
        ]
    },

    # ══════════════════════════════════════════════════════════
    #  ADDITIONAL SCENARIOS AND REASONING
    # ══════════════════════════════════════════════════════════
    {
        "name": "Scenario - ERL to IRL Cycle",
        "alias": [],
        "qa_pairs": [
            {
                "q": "Price just swept a major swing high on the H4, sweeping buy-side liquidity. What happens next?",
                "a": "After sweeping external range liquidity (the H4 swing high BSL), the ICT framework tells you price should now shift to internal range delivery — addressing the unfilled FVGs and untested Order Blocks within the range below. This is the ERL-to-IRL cycle. Step 1: Drop to M15 and watch for smart money reversal — displacement downward and MSS confirming the sweep was the objective. Step 2: If confirmed, look for a bearish OB or FVG on M15 for entry short. Step 3: Your targets are the internal range inefficiencies — unfilled bullish FVGs on the H1, untested Order Blocks, and eventually sell-side liquidity at the opposite extreme. This pattern repeats: ERL sweep → reversal → IRL fill → resume trend or seek opposite ERL."
            },
        ]
    },
    {
        "name": "Scenario - No Clear Bias",
        "alias": [],
        "qa_pairs": [
            {
                "q": "The market is choppy and I can't determine a clear bias. What should I do?",
                "a": "If you can't determine a clear daily bias, the ICT answer is simple: don't trade. No bias means no edge. Choppy markets happen when: the CBDR and Asian Range are both wide (already expanded, less directional energy remaining), multiple HTF PD arrays create conflicting signals, or price is consolidating between two significant liquidity pools without committing to either target. In this situation: 1) Step away from the lower timeframes — they'll only show noise. 2) Go to the Daily/H4 and identify the next significant BSL and SSL targets. 3) Wait for price to sweep ONE of them and show displacement. 4) Only then do you have a bias. Many of the best trading days come after 2-3 days of no-trade patience. Capital preservation IS profitable trading."
            },
        ]
    },
    {
        "name": "Scenario - Multi-Timeframe Alignment",
        "alias": [],
        "qa_pairs": [
            {
                "q": "Walk me through a complete multi-timeframe analysis for a trade.",
                "a": "Complete MTF analysis workflow: 1) WEEKLY: Identify the dealing range, is price in premium or discount? Where is the nearest BSL and SSL? This gives you the big picture. 2) DAILY: Read market structure (HH/HL or LH/LL). Check the last 20/40/60-day lookback for untested PD arrays. Determine if the weekly and daily biases agree. 3) H4: Identify the most recent Order Blocks and FVGs. Mark the immediate dealing range with premium/discount zones. This is your 'location' frame. 4) H1: Refine Kill Zone expectations. Where should the Judas Swing target? What PD arrays are in play? 5) M15: During the Kill Zone, watch for the liquidity sweep and MSS. This is your 'trigger' frame. 6) M5: Final entry execution — enter the OB/FVG created by the M15 displacement. Each timeframe adds precision: Weekly = direction, Daily = context, H4 = location, H1 = targets, M15 = trigger, M5 = execution."
            },
        ]
    },
    {
        "name": "Scenario - CBDR Expansion Setup",
        "alias": [],
        "qa_pairs": [
            {
                "q": "The CBDR was extremely tight at only 15 pips. What does this mean for my trading plan?",
                "a": "A 15-pip CBDR is exceptionally tight — this is a high-conviction expansion signal. Smart money spent 6 hours (2-8 PM EST) barely moving price, which means they were aggressively accumulating positions without tipping their hand. Your plan: 1) Expect a significant expansion move — likely 60+ pips based on SD projections from a 15-pip range. 2) Determine direction from Daily/H4 bias. 3) During the Asian session, watch for an equally tight range (confirming accumulation continues). 4) Mark the Asian Range and CBDR high/low as liquidity targets. 5) At London Open, expect a Judas Swing to sweep either the Asian or CBDR extreme. 6) After the sweep and MSS, enter with SD projections: -1.0 SD = 15 pips, -2.0 SD = 30 pips, -2.5 SD = 37.5 pips from the manipulation extreme. The tighter the CBDR, the more confident you can be in an explosive move."
            },
        ]
    },
    {
        "name": "Scenario - Order Block vs FVG Entry",
        "alias": [],
        "qa_pairs": [
            {
                "q": "Should I enter at the Order Block or the FVG when both are available?",
                "a": "When both an OB and FVG are present after displacement, the choice depends on your risk preference and model. The Order Block provides the more aggressive entry — it's further into the retracement, giving you a tighter stop and better R:R. But it carries more risk because price might reverse before reaching the OB. The FVG provides a more conservative entry — it's closer to current price and gets filled first, so higher probability of getting filled, but slightly worse R:R. The ideal solution: enter a partial position at the FVG, then add to it if price reaches the OB. This gives you exposure early while improving your average entry. If the FVG and OB overlap (Unicorn setup), that's the highest-conviction entry — enter the full position. In Model 12 scalping, FVG entries are preferred for speed. In Model 9, OB entries are preferred for precision."
            },
        ]
    },
    {
        "name": "Scenario - Session Transition",
        "alias": [],
        "qa_pairs": [
            {
                "q": "London pushed price up all session. Now NY is opening. Should I expect continuation or reversal?",
                "a": "The answer depends on whether London achieved its objective. If London's move swept a significant liquidity level AND reached a major PD array target, NY may reverse — London completed the day's work and NY provides the opposing move. If London's move was strong but hasn't reached the day's DOL (Draw on Liquidity target), NY AM typically continues the move — there's still unfinished institutional business. Check: 1) Did London sweep the targeted liquidity? 2) Is there still a significant PD array or liquidity pool ahead? 3) Is the H1/H4 OB or FVG that London was working toward already tested? If untouched targets remain, NY continues. If London hit the target, watch for NY manipulation to reverse. Also check SMT Divergence between sessions — if GBP/USD fails to confirm EUR/USD's London high, NY reversal probability increases."
            },
        ]
    },
]


def build_example(system_prompt: str, question: str, answer: str) -> dict:
    """Build a ChatML training example."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def generate_all_examples():
    """Generate all training examples from concept definitions."""
    examples = []
    for concept in CONCEPTS:
        for qa in concept["qa_pairs"]:
            ex = build_example(SYSTEM_PROMPT, qa["q"], qa["a"])
            examples.append(ex)
    return examples


def split_and_write(examples: list[dict]):
    """Split into train/valid/test and write JSONL files."""
    random.seed(42)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * 0.8)
    valid_end = int(n * 0.9)

    splits = {
        "train": examples[:train_end],
        "valid": examples[train_end:valid_end],
        "test": examples[valid_end:],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_examples in splits.items():
        path = OUTPUT_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name}: {len(split_examples)} examples -> {path}")

    return splits


def main():
    print("=" * 60)
    print("SYNTHETIC TRAINING DATA GENERATOR V2 — GAP FILLER")
    print("=" * 60)

    examples = generate_all_examples()
    print(f"\nGenerated {len(examples)} new Q&A pairs")
    print(f"From {len(CONCEPTS)} concepts\n")

    # Stats
    answers = [e["messages"][2]["content"] for e in examples]
    lengths = [len(a) for a in answers]
    print(f"Answer length stats:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Avg: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} chars")
    print()

    splits = split_and_write(examples)

    # Category breakdown
    categories = {}
    for concept in CONCEPTS:
        name = concept["name"]
        count = len(concept["qa_pairs"])
        categories[name] = count

    print(f"\nConcept/Q&A breakdown:")
    for name, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} Q&A pairs")

    print(f"\nTotal: {sum(len(v) for v in splits.values())} examples written to {OUTPUT_DIR}/")
    print(f"\nCombined with v3 data (44 examples), total coverage: {len(examples) + 44} examples")


if __name__ == "__main__":
    main()
