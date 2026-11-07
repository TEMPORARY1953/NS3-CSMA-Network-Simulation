// Assignment 3, variant 3. ns-3 (CMake-based releases, e.g. 3.40+).
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ns3;

namespace
{
const Time WINDOW = Seconds(10);
const Time DELAY = NanoSeconds(150); // 30 m / (2 * 10^8 m/s)
const DataRate RATE("100Mbps");
constexpr uint32_t PACKET_BYTES = 1500;
constexpr uint32_t QUEUE_PACKETS = 20;

struct QueueStat
{
    uint32_t size = 0;
    double last = 0;
    double area = 0;
};

struct Stats
{
    explicit Stats(uint32_t n) : queues(n) {}
    std::vector<QueueStat> queues;
    uint64_t generated = 0, queueDrops = 0, macDrops = 0, phyDrops = 0;
    uint64_t dequeued = 0, backoffs = 0, txBegin = 0, txEnd = 0;
    uint64_t generatedAt10 = 0, dropsAt10 = 0, txEndAt10 = 0, queuedAt10 = 0;
    uint32_t qMax = 0;
    double busy = 0;
    Time busyUntil = Seconds(0);

    void QueueChanged(uint32_t i, uint32_t oldSize, uint32_t newSize)
    {
        auto& q = queues.at(i);
        NS_ABORT_MSG_IF(q.size != oldSize, "Queue trace/accounting mismatch");
        const double t = std::min(Simulator::Now(), WINDOW).GetSeconds();
        q.area += q.size * (t - q.last);
        q.last = t;
        q.size = newSize;
        if (Simulator::Now() < WINDOW)
        {
            qMax = std::max(qMax, newSize);
        }
        NS_ABORT_MSG_IF(newSize > QUEUE_PACKETS, "Queue exceeded 20 packets");
    }

    void QueueDrop(Ptr<const Packet>) { ++queueDrops; }
    void MacDrop(Ptr<const Packet>) { ++macDrops; }
    void PhyDrop(Ptr<const Packet>) { ++phyDrops; }
    void Dequeue(Ptr<const Packet>) { ++dequeued; }
    void Backoff(Ptr<const Packet>) { ++backoffs; }
    void TxEnd(Ptr<const Packet>) { ++txEnd; }

    void TxBegin(Ptr<const Packet> p)
    {
        const Time now = Simulator::Now();
        NS_ABORT_MSG_IF(now < busyUntil, "Overlapping CSMA transmissions");
        ++txBegin;
        // Standard CsmaChannel is busy in TRANSMITTING and PROPAGATING.
        // These intervals do not overlap; IFG and waiting/backoff are excluded.
        busyUntil = now + RATE.CalculateBytesTxTime(p->GetSize()) + DELAY;
        if (now < WINDOW)
        {
            busy += (std::min(busyUntil, WINDOW) - now).GetSeconds();
        }
    }

    void Snapshot()
    {
        generatedAt10 = generated;
        dropsAt10 = macDrops + phyDrops;
        txEndAt10 = txEnd;
        for (auto& q : queues)
        {
            q.area += q.size * (WINDOW.GetSeconds() - q.last);
            q.last = WINDOW.GetSeconds();
            queuedAt10 += q.size;
        }
    }
};

void QueueChanged(Stats* stats, uint32_t i, uint32_t oldSize, uint32_t newSize)
{
    stats->QueueChanged(i, oldSize, newSize);
}

bool Receive(Ptr<NetDevice>, Ptr<const Packet>, uint16_t, const Address&)
{
    return true; // No replies and no IP/ARP traffic.
}

struct Source
{
    Ptr<NetDevice> device;
    Address destination;
    Ptr<RandomVariableStream> interval;
    Stats* stats;

    void ScheduleNext()
    {
        // Protect against rounding a very short random interval to zero ns.
        const Time dt = std::max(TimeStep(1), Seconds(interval->GetValue()));
        if (Simulator::Now() + dt < WINDOW)
        {
            Simulator::Schedule(dt, &Source::Generate, this);
        }
    }

    void Generate()
    {
        ++stats->generated; // Count creation BEFORE Send, even if Send fails.
        device->Send(Create<Packet>(PACKET_BYTES), destination, 0x88b5);
        ScheduleNext(); // Independent of completion, backoff, and queue drops.
    }
};
} // namespace

int main(int argc, char* argv[])
{
    uint32_t n = 10, seed = 12345;
    uint64_t run = 1;
    double alpha = 10;
    std::string dist = "exp";
    CommandLine cmd(__FILE__);
    cmd.AddValue("n", "Number of CSMA hosts", n);
    cmd.AddValue("alpha", "Generation rate per host, packets/s", alpha);
    cmd.AddValue("dist", "exp or uniform", dist);
    cmd.AddValue("seed", "RngSeedManager seed", seed);
    cmd.AddValue("run", "RngSeedManager run/substream", run);
    cmd.Parse(argc, argv);
    NS_ABORT_MSG_IF(n == 0 || !(alpha > 0) || seed == 0 || seed >= 4294944443ULL ||
                        (dist != "exp" && dist != "uniform"),
                    "Expected n>=1, alpha>0, valid positive seed, dist=exp|uniform");
    RngSeedManager::SetSeed(seed);
    RngSeedManager::SetRun(run);

    NodeContainer nodes;
    nodes.Create(n);
    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", DataRateValue(RATE));
    csma.SetChannelAttribute("Delay", TimeValue(DELAY));
    csma.SetDeviceAttribute("EncapsulationMode", StringValue("Dix"));
    csma.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", StringValue("20p"));
    NetDeviceContainer devices = csma.Install(nodes);
    // Keep standard backoff and interframe gap. Every host has distinct streams.
    csma.AssignStreams(devices, 1000000);

    Stats stats(n);
    std::vector<Source> sources(n); // Stable addresses for scheduled callbacks.
    for (uint32_t i = 0; i < n; ++i)
    {
        auto dev = DynamicCast<CsmaNetDevice>(devices.Get(i));
        dev->SetReceiveCallback(MakeCallback(&Receive));
        auto q = dev->GetQueue();
        bool connected = q->TraceConnectWithoutContext(
            "PacketsInQueue", MakeBoundCallback(&QueueChanged, &stats, i));
        connected &= q->TraceConnectWithoutContext(
            "Drop", MakeCallback(&Stats::QueueDrop, &stats));
        connected &= q->TraceConnectWithoutContext(
            "Dequeue", MakeCallback(&Stats::Dequeue, &stats));
        connected &= dev->TraceConnectWithoutContext(
            "MacTxDrop", MakeCallback(&Stats::MacDrop, &stats));
        connected &= dev->TraceConnectWithoutContext(
            "PhyTxDrop", MakeCallback(&Stats::PhyDrop, &stats));
        connected &= dev->TraceConnectWithoutContext(
            "MacTxBackoff", MakeCallback(&Stats::Backoff, &stats));
        connected &= dev->TraceConnectWithoutContext(
            "PhyTxBegin", MakeCallback(&Stats::TxBegin, &stats));
        connected &= dev->TraceConnectWithoutContext(
            "PhyTxEnd", MakeCallback(&Stats::TxEnd, &stats));
        NS_ABORT_MSG_IF(!connected, "Required trace source is unavailable");

        Ptr<RandomVariableStream> interval;
        if (dist == "exp")
        {
            auto rv = CreateObject<ExponentialRandomVariable>();
            rv->SetAttribute("Mean", DoubleValue(1.0 / alpha));
            rv->SetAttribute("Bound", DoubleValue(0));
            interval = rv;
        }
        else
        {
            auto rv = CreateObject<UniformRandomVariable>();
            rv->SetAttribute("Min", DoubleValue(0));
            rv->SetAttribute("Max", DoubleValue(2.0 / alpha));
            interval = rv;
        }
        interval->SetStream(i);
        sources[i] = {dev, devices.Get((i + 1) % n)->GetAddress(), interval, &stats};
        sources[i].ScheduleNext(); // First arrival is random too; no t=0 burst.
    }

    Simulator::Schedule(WINDOW, &Stats::Snapshot, &stats);
    // No Stop(10): drain all packets created in [0,10). No new arrivals after 10.
    Simulator::Run();
    const uint64_t lost = stats.macDrops + stats.phyDrops;
    const uint64_t admitted = stats.generated - stats.macDrops;
    const double area = std::accumulate(stats.queues.begin(), stats.queues.end(), 0.0,
                                      [](double a, const QueueStat& q) { return a + q.area; });
    for (const auto& q : stats.queues)
    {
        NS_ABORT_MSG_IF(q.size != 0, "Drain failed: queue is not empty");
    }
    NS_ABORT_MSG_IF(stats.generated != stats.generatedAt10 ||
                        stats.generated != lost + stats.txEnd ||
                        stats.queueDrops != stats.macDrops ||
                        admitted != stats.dequeued || stats.txBegin != stats.txEnd ||
                        stats.busy < 0 || stats.busy > WINDOW.GetSeconds() + 1e-8,
                    "Packet balance or busy-time check failed");

    // One machine-readable record per run. Counts include the fully drained cohort.
    std::cout << std::setprecision(17)
              << "{\"n\":" << n << ",\"alpha\":" << alpha
              << ",\"dist\":\"" << dist << "\",\"seed\":" << seed << ",\"run\":" << run
              << ",\"generated\":" << stats.generated << ",\"admitted\":" << admitted
              << ",\"queue_drops\":" << stats.queueDrops << ",\"phy_drops\":" << stats.phyDrops
              << ",\"lost\":" << lost << ",\"tx_end\":" << stats.txEnd
              << ",\"backoffs\":" << stats.backoffs
              << ",\"loss\":" << (stats.generated ? double(lost) / stats.generated : 0.0)
              << ",\"retries\":" << (admitted ? double(stats.backoffs) / admitted : 0.0)
              << ",\"q_mean\":" << area / (n * WINDOW.GetSeconds())
              << ",\"q_max\":" << stats.qMax
              << ",\"util\":" << stats.busy / WINDOW.GetSeconds()
              << ",\"queued_at_10\":" << stats.queuedAt10
              << ",\"unfinished_at_10\":" << stats.generatedAt10 - stats.dropsAt10 - stats.txEndAt10
              << ",\"tx_end_at_10\":" << stats.txEndAt10
              << ",\"drain_end_s\":" << Simulator::Now().GetSeconds() << "}\n";
    Simulator::Destroy();
    return 0;
}
