#!/usr/bin/env python3
"""
Generate high-quality synthetic training data for VEX ICT fine-tuning.

Instead of dumping graph triples as bullet lists, this creates natural-language
paragraph-style Q&A pairs that teach ICT concepts properly. Each answer weaves
the graph relationships into coherent explanations.

Target: 300-500 examples with 150-500 char answers.
"""
import json
import random
from pathlib import Path

OUTPUT_DIR = Path("finetune_data_v3")

SYSTEM_PROMPT = "You are VEX, an expert ICT (Inner Circle Trader) trading assistant. You help traders understand ICT methodology with clear, practical explanations."

# ============================================================
# ICT CONCEPT DEFINITIONS — written from graph + domain knowledge
# Each concept has: description, key_facts, and qa_pairs
# ============================================================

CONCEPTS = [
    {
        "name": "Fair Value Gap",
        "alias": ["FVG"],
        "description": "A Fair Value Gap (FVG) is a 3-candle pattern where candle 2's body creates a gap between the wicks of candles 1 and 3. This gap represents a price imbalance where one side of the market was dominant. FVGs act as magnets — price tends to return to fill these gaps before continuing. In ICT methodology, FVGs form after expansion moves and expose market imbalances that institutions created.",
        "qa_pairs": [
            {
                "q": "What is a Fair Value Gap and how do I trade it?",
                "a": "A Fair Value Gap (FVG) is a 3-candle pattern where candle 2's body creates a gap between the wicks of candles 1 and 3. This gap represents an area of price imbalance — one side of the market dominated so strongly that price left behind an inefficiency. FVGs form after expansion moves and expose market imbalances. To trade an FVG, wait for price to retrace back into the gap (the imbalance zone). Enter at the 50% level of the FVG with your stop beyond the opposite side. Target opposing liquidity pools. FVGs on higher timeframes (H4, Daily) carry more weight and are more reliable entry zones."
            },
            {
                "q": "How do I identify a Fair Value Gap on a chart?",
                "a": "To identify a Fair Value Gap, look for a 3-candle formation where the wicks of candle 1 and candle 3 do not overlap — there's a visible gap between them. Candle 2 is typically a large displacement candle that creates this imbalance. The FVG zone is the area between the high of candle 1's wick and the low of candle 3's wick (for bullish FVGs), or vice versa for bearish. Mark this zone on your chart — price will likely return to fill it. Unfilled FVGs below current price on higher timeframes are especially significant."
            },
            {
                "q": "What makes a Fair Value Gap high probability?",
                "a": "A high probability Fair Value Gap has several qualities: First, it forms during a displacement move with strong momentum — not just random chop. Second, it aligns with higher timeframe institutional order flow. Third, it occurs within a Kill Zone (London Open, NY AM, NY PM). Fourth, it hasn't been partially filled already. Fifth, and most importantly, it sits near a confluence with another PD Array element like an Order Block. The strongest FVGs are those created by institutional displacement where smart money is clearly accumulating or distributing."
            },
            {
                "q": "What is the difference between a Fair Value Gap and regular gaps?",
                "a": "A regular gap (like a weekend gap) is simply a price difference between close and open. A Fair Value Gap is a specific 3-candle intraday pattern where candle 2 creates an imbalance between candles 1 and 3 — the wicks don't overlap. FVGs occur within continuous trading sessions, not overnight. They represent areas where institutional order flow was so aggressive that price delivery was inefficient. While regular gaps may or may not fill, FVGs in ICT methodology are treated as magnets where price is drawn back to rebalance the inefficiency."
            },
        ]
    },
    {
        "name": "Order Block",
        "alias": ["OB"],
        "description": "An Order Block is the last opposite-colored candle before a strong displacement move. It marks where institutions placed large orders that initiated the move. Bullish OBs are the last bearish candle before a rally; bearish OBs are the last bullish candle before a sell-off.",
        "qa_pairs": [
            {
                "q": "What is an Order Block in ICT?",
                "a": "An Order Block is the last opposite-colored candle before a strong displacement move. It represents where institutions placed large orders that initiated a significant price move. A bullish Order Block is the last bearish (down-close) candle before price displaces upward. A bearish Order Block is the last bullish (up-close) candle before price displaces downward. When price returns to an Order Block, it often finds support or resistance there because institutional orders are still present at those levels. Order Blocks are one of the highest-priority elements in the PD Array Matrix."
            },
            {
                "q": "How do I identify and trade an Order Block?",
                "a": "To identify an Order Block, find a strong displacement move — a series of large candles moving aggressively in one direction. Then look backward to find the last opposing candle before that move began. That candle is your Order Block. To trade it: wait for price to retrace back to the Order Block zone (typically the body of that candle). Enter when price shows a reaction at the OB — a wick rejection or market structure shift on a lower timeframe. Place your stop beyond the Order Block's extreme. Target opposing liquidity. The strongest Order Blocks have displacement that creates a Fair Value Gap immediately after."
            },
            {
                "q": "What makes an Order Block valid vs invalid?",
                "a": "A valid Order Block must have displacement following it — strong, aggressive price movement away from the zone. Without displacement, it's just a random candle. An OB becomes invalid (or 'broken') when price trades through it completely on a closing basis. Key validation criteria: the displacement should create at least one Fair Value Gap, confirming institutional involvement. The Order Block should align with higher timeframe bias. It should occur during a Kill Zone for maximum probability. An OB that has been tested once and held is stronger than an untested one. Order Blocks on higher timeframes (H1, H4, Daily) carry more institutional weight."
            },
        ]
    },
    {
        "name": "Liquidity",
        "alias": ["BSL", "SSL"],
        "description": "Liquidity in ICT refers to clusters of stop-loss orders resting at predictable levels — above swing highs (buy-side liquidity/BSL) and below swing lows (sell-side liquidity/SSL). Smart money targets these pools to fill large orders.",
        "qa_pairs": [
            {
                "q": "What is liquidity in ICT methodology?",
                "a": "In ICT methodology, liquidity refers to clusters of stop-loss orders resting at predictable price levels. Buy-side liquidity (BSL) sits above swing highs and equal highs, where buy stops and breakout orders accumulate. Sell-side liquidity (SSL) sits below swing lows and equal lows, where sell stops rest. Smart money needs this liquidity to fill their large institutional orders — they can't enter massive positions without sufficient counterparty volume. Understanding where liquidity rests is fundamental to ICT because price is drawn to these pools before reversals. The market is essentially an engine that seeks liquidity."
            },
            {
                "q": "How do I identify buy-side and sell-side liquidity?",
                "a": "Buy-side liquidity forms above swing highs, equal highs, and obvious resistance levels where retail traders place buy stops and breakout entries. On your chart, mark every obvious swing high — those are liquidity targets. Equal highs are especially potent because double tops create concentrated stop clusters. Sell-side liquidity forms below swing lows, equal lows, and obvious support levels. Equal lows are powerful SSL targets. To use this: ask yourself 'where are the stops?' Price will move toward those stops. Once liquidity is swept (stops triggered), price often reverses. This is the foundation of the Judas Swing and manipulation concepts."
            },
            {
                "q": "Why does price always seek liquidity?",
                "a": "Price seeks liquidity because institutional traders need counterparty volume to fill their large orders. If a bank wants to buy 500 million EUR/USD, they can't just hit 'buy' — that would move price against them. Instead, they engineer moves that trigger stop losses and breakout entries at predictable levels, creating the volume they need. This is why price often spikes above obvious highs (sweeping buy-side liquidity) before reversing, or dips below obvious lows (sweeping sell-side liquidity) before rallying. Understanding this is what separates ICT from retail trading — you stop being the liquidity and start trading with the institutions."
            },
        ]
    },
    {
        "name": "Displacement",
        "alias": [],
        "description": "Displacement is a strong, aggressive price move characterized by large institutional candles. It confirms directional bias, often creates Fair Value Gaps and Order Blocks, and validates that smart money is committed to a direction.",
        "qa_pairs": [
            {
                "q": "What is displacement in ICT?",
                "a": "Displacement is a strong, aggressive price move created by institutional order flow. It's characterized by one or more large candles moving decisively in one direction, often accompanied by increased volume. Displacement is critical because it confirms that smart money is committed to a direction — it establishes directional bias. A valid displacement creates Fair Value Gaps and leaves behind Order Blocks. Without displacement, there's no confirmation that institutions are involved. You should never enter a trade without seeing displacement first — it's the proof that the price move has institutional sponsorship behind it."
            },
            {
                "q": "How do I confirm displacement on the chart?",
                "a": "Displacement shows up as one or more large candles moving aggressively away from a level. Key confirmation signals: First, the candle bodies are significantly larger than recent average candles — this isn't normal price action. Second, the move creates at least one Fair Value Gap (3-candle imbalance). Third, it breaks through a recent swing point or structure level. Fourth, it happens during optimal times — Kill Zones like London Open or NY AM session. Bearish displacement shows multiple large bearish candles confirming sell pressure. Bullish displacement shows the opposite. The displacement must precede any valid Order Block identification."
            },
        ]
    },
    {
        "name": "Market Structure",
        "alias": ["MSS", "Market Structure Shift"],
        "description": "Market structure in ICT refers to the pattern of higher highs/higher lows (bullish) or lower highs/lower lows (bearish). A Market Structure Shift (MSS) signals a potential reversal when the pattern breaks.",
        "qa_pairs": [
            {
                "q": "What is market structure in ICT?",
                "a": "Market structure in ICT refers to the swing point pattern that defines trend direction. Bullish market structure shows higher highs (HH) and higher lows (HL) — each swing point is higher than the last. Bearish market structure shows lower highs (LH) and lower lows (LL). Understanding market structure tells you who is in control — bulls or bears. It represents the Smart Money footprint showing institutional accumulation (bullish) or distribution (bearish). Always read market structure from the higher timeframe first, then drill down to lower timeframes for entry timing."
            },
            {
                "q": "What is a Market Structure Shift and why does it matter?",
                "a": "A Market Structure Shift (MSS) occurs when price rapidly reverses and breaks a short-term swing point, signaling the beginning of a new trend or reversal. In bearish structure (LH/LL), an MSS happens when price breaks above a recent lower high with displacement. In bullish structure (HH/HL), an MSS happens when price breaks below a recent higher low. This matters because it signals that institutional sponsorship has changed direction — smart money is now on the other side. An MSS is your confirmation to look for entries in the new direction. Always wait for the MSS before taking reversal trades."
            },
            {
                "q": "How do I read market structure across multiple timeframes?",
                "a": "Start with the higher timeframe (Daily or H4) to determine the overall bias — is it making HH/HL (bullish) or LH/LL (bearish)? This gives you the direction you want to trade. Then move to the H1 to identify key levels — Order Blocks, FVGs, and liquidity pools that align with the HTF bias. Finally, drop to M15 or M5 for entry timing. The key rule: the lower timeframe must align with the higher timeframe view via a Market Structure Shift before you enter. If HTF is bullish but LTF is still bearish, wait. Don't enter until the LTF shifts to confirm the HTF view."
            },
        ]
    },
    {
        "name": "Kill Zones",
        "alias": ["Killzone", "KZ"],
        "description": "Kill Zones are specific time windows when institutional trading volume is highest and price moves are most reliable. The main Kill Zones are Asian (8PM-12AM EST), London Open (2AM-5AM EST), NY AM (8:30AM-11AM EST), and NY PM (1:30PM-4PM EST).",
        "qa_pairs": [
            {
                "q": "What are Kill Zones in ICT?",
                "a": "Kill Zones are specific time windows during the trading day when institutional activity is highest and price moves are most reliable. There are four main Kill Zones: the Asian session (8:00 PM - 12:00 AM EST), London Open (2:00 AM - 5:00 AM EST), New York AM (8:30 AM - 11:00 AM EST), and New York PM (1:30 PM - 4:00 PM EST). Smart money operates during these windows because that's when volume and volatility peak. Trading outside Kill Zones dramatically reduces your probability. The London and NY AM sessions are typically the highest probability Kill Zones for ICT setups."
            },
            {
                "q": "Why should I only trade during Kill Zones?",
                "a": "Trading during Kill Zones matters because institutional order flow is concentrated in these windows. Outside Kill Zones, price action is often just noise — random movement without institutional direction. When you trade during Kill Zones, you align with the times when smart money is active, which means Order Blocks and FVGs are more likely to be respected, displacement moves are genuine, and setups have institutional sponsorship. The Kill Zones also align with major economic data releases and session opens, which create the volatility and liquidity that make ICT setups work. Trading outside these times is gambling."
            },
        ]
    },
    {
        "name": "Judas Swing",
        "alias": [],
        "description": "A Judas Swing is a false breakout at the start of a trading session that lures retail traders in the wrong direction before price reverses to the true trend direction.",
        "qa_pairs": [
            {
                "q": "What is a Judas Swing?",
                "a": "A Judas Swing is a false move at the beginning of a trading session — typically during London Open or New York Open — that tricks retail traders into taking positions in the wrong direction. It's named after the biblical betrayal because price 'lies' about its true intention. The Judas Swing sweeps liquidity (triggers stops) on one side before reversing aggressively in the true direction. For example, price might spike above the Asian session high, triggering buy stops, then reverse sharply downward. This false breakout traps traders on the wrong side while giving smart money the liquidity they need to fill their real orders."
            },
            {
                "q": "How do I identify and trade a Judas Swing?",
                "a": "A Judas Swing typically occurs during the first few hours of the London or New York session. Watch for price to make a false move in the opposite direction of the anticipated daily bias — usually sweeping the Asian session high or low. The key signs: price quickly moves to take out an obvious liquidity level, then stalls and shows displacement in the opposite direction with a Market Structure Shift. To trade it, don't chase the initial move. Wait for the reversal displacement and MSS, then enter on a retracement to an Order Block or FVG created during the reversal. Set stops near the extreme of the initial false swing."
            },
        ]
    },
    {
        "name": "CBDR",
        "alias": ["Central Bank Dealers Range"],
        "description": "The Central Bank Dealers Range (CBDR) is measured from 2:00 PM to 8:00 PM EST (prior day). When the CBDR is tight (under 40 pips), it signals that smart money is accumulating positions and a large expansion move is coming.",
        "qa_pairs": [
            {
                "q": "What is CBDR and how do I use it?",
                "a": "CBDR (Central Bank Dealers Range) is the price range established between 2:00 PM and 8:00 PM EST the prior trading day. It represents the period when central bank dealers and institutional market makers position themselves before the next session's move. To use it: measure the high-to-low range of those 6 hours. If the CBDR is tight (ideally under 40 pips on EUR/USD), it signals accumulation — smart money is building positions without moving price much, and a significant expansion is coming in the next London/NY session. Mark the CBDR high and low, then apply standard deviation extensions to project expansion targets."
            },
            {
                "q": "What does a tight CBDR tell me about the next session?",
                "a": "A tight CBDR (under 40 pips) tells you that smart money was quietly accumulating positions during the 2-8 PM EST window. When big players build positions without moving price much, it means they're setting up for a large expansion move. Think of it like a coiled spring — the tighter the CBDR, the bigger the expected expansion. Compare your CBDR measurement to the Asian Range as well. When both are tight, the London or NY AM session is very likely to produce a strong directional move. Use the CBDR's standard deviation projections to set your expansion targets."
            },
        ]
    },
    {
        "name": "Smart Money",
        "alias": ["Institutional Traders", "Smart Money Concepts", "SMC"],
        "description": "Smart Money refers to institutional traders — banks, hedge funds, and central bank dealers — who move the markets. They follow structured rules, accumulate when CBDR is tight, and use liquidity pools to fill their large orders.",
        "qa_pairs": [
            {
                "q": "What is Smart Money in ICT?",
                "a": "Smart Money in ICT refers to institutional market participants — central banks, major banks, hedge funds, and large prop firms — who have the capital to actually move markets. Unlike retail traders, smart money follows structured rules based on time (Kill Zones), price levels (Order Blocks, FVGs), and liquidity. They accumulate positions when the CBDR is tight, sell into buy-side liquidity, and buy into sell-side liquidity — always on the 'left side of the curve.' Understanding how smart money operates is the core of ICT methodology: you learn to read their footprints in price action and position yourself to trade alongside them, not against them."
            },
            {
                "q": "How does Smart Money enter and exit positions?",
                "a": "Smart money can't just hit buy or sell on their platform — their order sizes are too large. They need counterparty liquidity. So they engineer price moves that create this liquidity. During accumulation, they build positions quietly during tight CBDR periods. They create false moves (Judas Swings) that sweep liquidity pools, triggering retail stops and breakout orders to fill their real positions. They enter at Order Blocks and use displacement to move price away from their entry. They distribute (exit) during expansion phases, selling into buy-side liquidity as price moves in their favor. Every ICT concept traces back to this fundamental dynamic."
            },
        ]
    },
    {
        "name": "Power of Three",
        "alias": ["PO3", "AMD"],
        "description": "The Power of Three (PO3) describes the three phases every trading session follows: Accumulation, Manipulation, Distribution. Also called AMD.",
        "qa_pairs": [
            {
                "q": "What is the Power of Three in ICT?",
                "a": "The Power of Three (PO3), also called AMD, describes the three phases that every trading session follows. Phase 1 is Accumulation — smart money quietly builds positions during low-volatility periods (like the Asian session or tight CBDR). Phase 2 is Manipulation — a false move (Judas Swing) that sweeps liquidity and traps retail traders on the wrong side. Phase 3 is Distribution — the real move where smart money drives price in the intended direction, distributing their position as momentum traders jump in. Understanding PO3 gives you the blueprint for the entire trading day and helps you avoid being trapped during the manipulation phase."
            },
            {
                "q": "How does the Power of Three play out during a trading day?",
                "a": "During a typical trading day, PO3 unfolds across sessions. Accumulation happens during the Asian session and CBDR period — price consolidates in a tight range while smart money builds positions. Manipulation occurs at or near the London Open — price makes a false move (Judas Swing) to sweep the Asian session high or low, triggering stops and creating liquidity. Distribution is the real move that typically happens during the London session or NY AM — price displaces aggressively in the true direction, moving toward the day's objective (opposing liquidity pool). Your job is to recognize which phase you're in and only enter during late manipulation or early distribution."
            },
        ]
    },
    {
        "name": "PD Array Matrix",
        "alias": ["Premium Discount Array"],
        "description": "The PD Array Matrix is the priority hierarchy of institutional reference points used for entries. In order: Order Blocks, Fair Value Gaps, Breaker Blocks, Mitigation Blocks, Liquidity Voids, NWOG/NDOG.",
        "qa_pairs": [
            {
                "q": "What is the PD Array Matrix?",
                "a": "The PD Array Matrix is ICT's priority hierarchy of institutional reference points used to identify high-probability trade entries. The components in priority order are: 1) Order Blocks — the highest priority entry zones, 2) Fair Value Gaps — price imbalances acting as magnets, 3) Breaker Blocks — failed order blocks that flip from support to resistance or vice versa, 4) Mitigation Blocks — levels left behind during initial moves, 5) Liquidity Voids — large candles with no opposing activity, 6) NWOG/NDOG (New Week/Day Opening Gaps) — session opening imbalances. When multiple PD Array elements converge at the same level, that creates a powerful confluence zone for entries."
            },
        ]
    },
    {
        "name": "SMT Divergence",
        "alias": ["Smart Money Technique Divergence"],
        "description": "SMT Divergence occurs when correlated instruments make non-congruent price action — one makes a new high/low while the other fails to confirm it, signaling smart money activity.",
        "qa_pairs": [
            {
                "q": "What is SMT Divergence and how do I use it?",
                "a": "SMT Divergence (Smart Money Technique Divergence) occurs when two correlated instruments show non-congruent price action — one makes a new swing high or low while the other fails to confirm. For example, if EUR/USD makes a new low but GBP/USD fails to make a corresponding new low, that's SMT Divergence signaling that the move down is losing institutional conviction. SMT Divergence is best used for confirmation — after a liquidity sweep, during Kill Zones, or to confirm a Market Structure Shift. It scores +15 points in setup evaluation. Important: SMT Divergence alone is not an entry signal. Use it alongside other confluence factors like Order Blocks and FVGs."
            },
        ]
    },
    {
        "name": "Market Maker Sell Model",
        "alias": ["MMSM"],
        "description": "A blueprint for bearish price delivery with four stages: Original Consolidation (accumulation), Engineering Liquidity (creating buy-side traps), Smart Money Reversal (sweeping BSL), and Distribution (selling down to targets).",
        "qa_pairs": [
            {
                "q": "What is the Market Maker Sell Model?",
                "a": "The Market Maker Sell Model (MMSM) is ICT's blueprint for bearish price delivery from bullish PD arrays to bearish ones. It has four stages: Stage 1 — Original Consolidation, where price ranges between two limits while smart money builds short positions and retail accumulates longs at support. Stage 2 — Engineering Liquidity, where price makes higher lows creating buy-side liquidity; each higher low becomes stops for shorts, setting the trap. Stage 3 — Smart Money Reversal, where price spikes above consolidation highs, sweeps buy-side liquidity (BSL), and reverses. Stage 4 — Distribution, where price sells off targeting sell-side liquidity below old lows (the Terminus). The model helps you identify the stages in real-time so you can position for the reversal."
            },
        ]
    },
    {
        "name": "Market Maker Buy Model",
        "alias": ["MMBM"],
        "description": "The bullish counterpart of the MMSM. Price consolidates, engineers sell-side liquidity, sweeps SSL, then rallies to buy-side targets.",
        "qa_pairs": [
            {
                "q": "What is the Market Maker Buy Model?",
                "a": "The Market Maker Buy Model (MMBM) is the bullish counterpart to the Market Maker Sell Model. It describes how institutions engineer a bullish reversal. Price first consolidates (accumulation), then trades down to create distribution — sweeping sell-side liquidity below old lows to fill large buy orders. After the liquidity sweep, smart money drives price up through displacement, targeting buy-side liquidity above old highs. The model consists of: consolidation, engineering sell-side liquidity (creating lower lows that attract short sellers), the reversal sweep (grabbing SSL), and the rally up. Recognizing this pattern helps you buy at the bottom of the engineered move rather than selling into the trap."
            },
        ]
    },
    {
        "name": "Optimal Trade Entry",
        "alias": ["OTE"],
        "description": "The Optimal Trade Entry is the 61.8% to 79% Fibonacci retracement zone within an expansion leg, providing the best risk:reward entry point.",
        "qa_pairs": [
            {
                "q": "What is Optimal Trade Entry in ICT?",
                "a": "Optimal Trade Entry (OTE) is the 61.8% to 79% Fibonacci retracement zone within a displacement leg. After a strong move (displacement), price often retraces to this zone before continuing in the original direction. The OTE gives you the best risk:reward ratio because you're entering deep into the retracement — close to where your stop should be — while maintaining a target at or beyond the swing extreme. To use it: after displacement creates a new swing, draw Fibonacci from the swing low to swing high (bullish) or high to low (bearish). The 61.8%-79% zone is your OTE. Look for Order Blocks or FVGs within this zone for precise entries."
            },
        ]
    },
    {
        "name": "Breaker Block",
        "alias": [],
        "description": "A Breaker Block is a failed Order Block — an OB that was broken through, causing a market structure shift. The broken OB then acts as support/resistance from the other side.",
        "qa_pairs": [
            {
                "q": "What is a Breaker Block?",
                "a": "A Breaker Block is a failed Order Block. When price breaks through an Order Block instead of respecting it, that OB becomes a Breaker Block on the other side. For example, if a bullish Order Block fails and price breaks through it to the downside, that zone becomes a bearish Breaker Block — it now acts as resistance if price returns to it. Breaker Blocks are significant because they represent a shift in market structure. They signal that institutional commitment has changed direction. In the PD Array Matrix, Breaker Blocks rank third in priority after Order Blocks and Fair Value Gaps. They serve as strong support or resistance levels where price often reacts on a retest."
            },
        ]
    },
    {
        "name": "Mitigation Block",
        "alias": [],
        "description": "A Mitigation Block is the candle that caused a false move or liquidity raid. When price returns, institutions mitigate their losing position at that level.",
        "qa_pairs": [
            {
                "q": "What is a Mitigation Block?",
                "a": "A Mitigation Block is identified as the last significant move before price reversed sharply — specifically, the candle or zone that caused a false move or liquidity raid. When price returns to this level, institutions mitigate (close out) their losing positions from the initial false move. This creates a reaction in price, making it a tradeable level. Mitigation Blocks differ from Order Blocks because they represent institutional loss mitigation, not initial order placement. They're created by sharp moves that left behind untested levels. In the PD Array Matrix, Mitigation Blocks rank fourth, after Order Blocks, FVGs, and Breaker Blocks."
            },
        ]
    },
    {
        "name": "Silver Bullet",
        "alias": [],
        "description": "The Silver Bullet is an ICT setup that targets FVGs formed during specific 1-hour precision windows within each Kill Zone for high-probability entries.",
        "qa_pairs": [
            {
                "q": "What is the Silver Bullet setup in ICT?",
                "a": "The Silver Bullet is a precision trade setup that targets Fair Value Gaps formed during specific 1-hour windows within each Kill Zone. The concept exploits the idea that within broader Kill Zone windows, there are even more specific 60-minute periods when institutional activity creates the most reliable FVGs. During these windows, look for displacement that creates an FVG, then enter when price retraces to the FVG with your stop beyond the FVG and target at opposing liquidity. The Silver Bullet is about surgical precision — waiting for the exact right window, the exact right FVG, and executing with tight risk. It applies the core ICT principles within a compressed time frame."
            },
        ]
    },
    {
        "name": "Asian Range",
        "alias": [],
        "description": "The Asian Range is established between 8:00 PM and 12:00 AM EST. Its high and low often get swept during London Open as part of the daily manipulation phase.",
        "qa_pairs": [
            {
                "q": "What is the Asian Range and why does it matter?",
                "a": "The Asian Range is the price range established between 8:00 PM and 12:00 AM EST. It matters because this range typically represents the accumulation phase of the Power of Three (PO3). Smart money builds positions during this quiet period, creating a range with clearly defined highs and lows. These highs and lows become liquidity targets for the next session. During London Open, price will often sweep the Asian Range high or low — this is the manipulation phase. If the daily bias is bullish, expect the Asian low to be taken (sweeping sell-side liquidity) before price rallies. If bearish, expect the Asian high to be swept before the sell-off."
            },
        ]
    },
    {
        "name": "Institutional Order Flow",
        "alias": ["IOF"],
        "description": "Institutional Order Flow refers to the dominant direction smart money pushes price. It follows higher timeframe direction and influences market structure shifts.",
        "qa_pairs": [
            {
                "q": "What is Institutional Order Flow?",
                "a": "Institutional Order Flow (IOF) refers to the dominant direction that smart money is pushing price. It's not visible on any indicator — you read it through market structure, displacement, and how price interacts with Order Blocks and FVGs. IOF follows the higher timeframe directional bias established on the Daily and Weekly charts. When IOF is bullish, price makes higher lows that get respected and higher highs that get broken. When IOF is bearish, higher highs get rejected and lower lows get targeted. Understanding IOF means understanding the 'why' behind gap creation and price movement — it's the institutional sponsorship model that drives everything in ICT."
            },
        ]
    },
    {
        "name": "Liquidity Pools",
        "alias": [],
        "description": "Liquidity Pools are specific areas where stop-loss orders concentrate — above swing highs (BSL) and below swing lows (SSL). Equal highs/lows create the most concentrated pools.",
        "qa_pairs": [
            {
                "q": "What are liquidity pools and how are they targeted?",
                "a": "Liquidity pools are specific price areas where stop-loss and pending orders concentrate. Buy-side liquidity pools form above swing highs, equal highs, and obvious resistance levels — these contain buy stops from short sellers and breakout buy orders from trend followers. Sell-side liquidity pools form below swing lows, equal lows, and obvious support levels. Equal highs and equal lows create the most concentrated pools because they look like strong support/resistance to retail traders, who stack their stops just beyond them. Institutions target these pools because they provide the counterparty volume needed to fill large orders. When a pool is captured (swept), price often reverses — the Liquidity Run Model describes this targeting process."
            },
        ]
    },
    {
        "name": "Flash Wick",
        "alias": [],
        "description": "A Flash Wick is a rapid price spike (often ~20 pips in 2 minutes) that quickly reverses — representing either a Judas manipulation or a liquidity grab pattern.",
        "qa_pairs": [
            {
                "q": "What is a Flash Wick in ICT?",
                "a": "A Flash Wick is a rapid price spike — often around 20 pips in just 2 minutes — that quickly reverses. It represents an aggressive liquidity grab where price shoots to a key level, triggers clustered stop-loss orders, then immediately snaps back. Flash Wicks can be classified as either Judas manipulation (when they occur at session opens and represent false directional signals) or reversal signals (when they sweep a key liquidity level and price has reached its ultimate objective). They're closely related to the Judas Swing concept but are more extreme and compressed in time. When you see a flash wick at a significant level during a Kill Zone, it often signals that liquidity has been captured and price may now reverse."
            },
        ]
    },
    {
        "name": "Manipulation",
        "alias": [],
        "description": "Manipulation in ICT is the engineered false move that sweeps liquidity. It includes stop hunts, fake breakouts, Judas Swings, and occurs during specific times or around significant economic events.",
        "qa_pairs": [
            {
                "q": "What is manipulation in ICT methodology?",
                "a": "Manipulation in ICT is the engineered phase of price delivery where smart money creates false moves to sweep liquidity and trap retail traders. It includes stop hunts, fake breakouts, Judas Swings, and spikes below sell-side liquidity. Manipulation typically occurs during specific times — particularly around London Open (2:00-5:00 AM EST) and around significant economic events. The key tell: manipulation creates a false move in the opposite direction of the true daily bias. It involves techniques like spiking below SSL to fill buy orders, or above BSL to fill sells. Understanding manipulation is Phase 2 of the Power of Three — between accumulation and distribution. Your goal is to recognize manipulation in real-time and position yourself for the distribution phase."
            },
        ]
    },
    # Additional scenario/reasoning pairs
    {
        "name": "Trade Scenarios",
        "alias": [],
        "description": "",
        "qa_pairs": [
            {
                "q": "Price is at a Daily bullish Order Block and the M15 just showed a Market Structure Shift bullish. The CBDR was tight at 18 pips. What should I do?",
                "a": "This is a high-probability long setup. Here's why: The Daily bullish Order Block gives you higher timeframe location — you're at an institutional reference point. The M15 MSS bullish confirms that lower timeframe structure has shifted to align with the HTF bias. The tight 18-pip CBDR signals institutional accumulation — smart money was building positions quietly. You should look for an entry on the M5, targeting the M15 FVG or Order Block created by the MSS. Place your stop below the M15 swing low that created the MSS. Target buy-side liquidity above recent swing highs. This setup has strong confluence."
            },
            {
                "q": "I see a bearish FVG on the H4 but the Daily trend is bullish. Should I short?",
                "a": "No — this is a counter-trend trade with poor confluence. In ICT methodology, the higher timeframe always takes priority. If the Daily trend is bullish (making HH/HL), a bearish H4 FVG is likely just a retracement within the larger bullish move. Instead of shorting, use that H4 FVG as context — if price fills it and shows bullish displacement with an MSS on the M15, that's actually a buy setup. You'd be entering long at a discount (the H4 FVG fill) in alignment with the Daily bullish bias. The rule: HTF provides location and direction, LTF provides entry timing. Never trade against the HTF."
            },
            {
                "q": "It's 3 PM EST and I see a great Order Block setup. Should I take it?",
                "a": "No — 3:00 PM EST is at the tail end of the NY PM Kill Zone and approaching the close of institutional activity for the day. The highest probability trades occur during London Open (2-5 AM EST) and NY AM (8:30-11 AM EST). Taking setups at 3 PM means you're trading when institutional volume is declining and the day's primary moves have likely already occurred. Additionally, the 2:00-8:00 PM window is the CBDR period — this is when smart money is accumulating for the NEXT session's move, not creating tradeable setups. Wait for tomorrow's Kill Zone instead."
            },
            {
                "q": "Price swept the Asian Range low, showed bullish displacement, and created an FVG during London Open. Walk me through the trade.",
                "a": "This is a textbook bullish Power of Three setup. Phase 1 (Accumulation): Price consolidated during the Asian session, creating the range. Phase 2 (Manipulation): Price swept the Asian Range low — this is the Judas Swing taking sell-side liquidity below the Asian low, triggering retail sell stops. Phase 3 (Distribution): Bullish displacement confirms smart money has reversed from the liquidity sweep. The FVG created by the displacement is your entry zone. Wait for price to retrace into the FVG. Enter long at or near the 50% level of the FVG. Place your stop below the sweep low (the Asian Range low). Target buy-side liquidity above the Asian Range high or higher. This setup has Kill Zone timing (London Open), institutional liquidity grab, displacement confirmation, and a clean FVG entry."
            },
            {
                "q": "How do I determine my daily bias before the session starts?",
                "a": "Daily bias determination starts with higher timeframe analysis the night before. Check the Daily and H4 chart for market structure — is it making HH/HL (bullish) or LH/LL (bearish)? Identify the nearest buy-side and sell-side liquidity targets. Then measure the CBDR (2-8 PM EST range). A tight CBDR signals expansion is coming. Check if the previous day left any unfilled FVGs or Order Blocks — price may target these. Look at the Asian Range for accumulation context. Consider correlation: if EUR/USD and GBP/USD are both showing similar bias, that strengthens your conviction. Finally, check for any high-impact news events that could be the catalyst. Your bias should answer: which direction are the institutions likely to push price today, and what liquidity target are they aiming for?"
            },
            {
                "q": "What are the most common mistakes ICT traders make?",
                "a": "The most common ICT mistakes are: First, trading outside Kill Zones — taking setups at random times when institutional volume is low. Second, counter-trend trading — fighting the higher timeframe bias because a lower timeframe setup looks good. Third, entering before Market Structure Shift confirmation — jumping in on anticipation rather than waiting for the LTF to confirm the HTF view. Fourth, treating FVGs and Order Blocks as automatic entries — these are reference points, not signals. You need displacement and MSS confirmation. Fifth, ignoring the CBDR — a wide CBDR signals low probability expansion. Sixth, over-trading — taking every setup instead of waiting for A+ confluence with multiple ICT factors aligning. Seventh, not understanding liquidity — failing to identify where stops are clustered and who is being trapped."
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
    # 80/10/10 split
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
    print("SYNTHETIC TRAINING DATA GENERATOR v3")
    print("=" * 60)

    examples = generate_all_examples()
    print(f"\nGenerated {len(examples)} high-quality Q&A pairs")
    print(f"From {len(CONCEPTS)} ICT concepts\n")

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

    print(f"\nTotal: {sum(len(v) for v in splits.values())} examples written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
