#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#define private public
#include "diepcustom/headless.hpp"
#undef private

using diepcustom::headless::Action;
using diepcustom::headless::Config;
using diepcustom::headless::Simulation;

namespace {
constexpr int EpisodeStatsFieldCount = 16;
constexpr int CombatChannelCount = 18;
constexpr int DamageDealtIndex = 4;
constexpr int EnemyDamageDealtIndex = 5;
constexpr int ShotsFiredIndex = 7;
constexpr int EnemyKillsIndex = 9;
constexpr int FarmKillsIndex = 10;
constexpr int DeathCountIndex = 11;
constexpr int UpgradeChoicesIndex = 15;

struct ParsedEntity {
  std::string kind;
  double x = 0;
  double y = 0;
  double size = 0;
};

double extractNumber(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\":(-?[0-9]+(?:\\.[0-9]+)?)");
  std::smatch match;
  assert(std::regex_search(json, match, pattern));
  return std::stod(match[1].str());
}

std::vector<ParsedEntity> parseEntities(const std::string& json) {
  std::vector<ParsedEntity> entities;
  const std::regex pattern("\\{\"id\":[0-9]+.*?\"kind\":\"([^\"]+)\".*?\"position\":\\{\"x\":(-?[0-9]+(?:\\.[0-9]+)?),\"y\":(-?[0-9]+(?:\\.[0-9]+)?),\"angle\":-?[0-9]+(?:\\.[0-9]+)?\\}.*?\"physics\":\\{\"sides\":[0-9]+,\"size\":(-?[0-9]+(?:\\.[0-9]+)?)");
  for (std::sregex_iterator it(json.begin(), json.end(), pattern), end; it != end; ++it) {
    entities.push_back(ParsedEntity{(*it)[1].str(), std::stod((*it)[2].str()), std::stod((*it)[3].str()), std::stod((*it)[4].str())});
  }
  return entities;
}

int countFarmables(const std::vector<ParsedEntity>& entities) {
  return static_cast<int>(std::count_if(entities.begin(), entities.end(), [](const ParsedEntity& entity) {
    return entity.kind == "shape" || entity.kind == "crasher";
  }));
}

void assertNoAgentFarmableOverlap(const std::vector<ParsedEntity>& entities) {
  for (const auto& agent : entities) {
    if (agent.kind != "agent") continue;
    for (const auto& farmable : entities) {
      if (!(farmable.kind == "shape" || farmable.kind == "crasher")) continue;
      const double dx = agent.x - farmable.x;
      const double dy = agent.y - farmable.y;
      const double minDistance = agent.size + farmable.size;
      assert(dx * dx + dy * dy > minDistance * minDistance);
    }
  }
}

void assertTrainingScenario(const std::string& scenario, double arenaSize, int targetShapes) {
  Simulation sim(Config{777, 8, 128, scenario});
  const std::string snapshot = sim.fullWorldSnapshotJson();
  assert(extractNumber(snapshot, "leftX") == -arenaSize * 0.5);
  assert(extractNumber(snapshot, "rightX") == arenaSize * 0.5);
  assert(extractNumber(snapshot, "topY") == -arenaSize * 0.5);
  assert(extractNumber(snapshot, "bottomY") == arenaSize * 0.5);

  const auto entities = parseEntities(snapshot);
  assert(static_cast<int>(std::count_if(entities.begin(), entities.end(), [](const ParsedEntity& entity) { return entity.kind == "agent"; })) == 8);
  assert(countFarmables(entities) >= targetShapes * 95 / 100);
  assert(countFarmables(entities) <= targetShapes);
  for (const auto& entity : entities) {
    assert(entity.x >= -arenaSize * 0.5);
    assert(entity.x <= arenaSize * 0.5);
    assert(entity.y >= -arenaSize * 0.5);
    assert(entity.y <= arenaSize * 0.5);
  }
  assertNoAgentFarmableOverlap(entities);

  Simulation same(Config{777, 8, 128, scenario});
  Simulation different(Config{778, 8, 128, scenario});
  assert(snapshot == same.fullWorldSnapshotJson());
  assert(snapshot != different.fullWorldSnapshotJson());

  int removed = 0;
  for (auto& entity : sim.entities_) {
    if (entity.kind == "shape" || entity.kind == "crasher") {
      entity.removed = true;
      removed += 1;
      if (removed == 40) break;
    }
  }
  sim.cleanupEntities();
  assert(sim.countLiveTrainingShapes() <= targetShapes - 40);
  for (int i = 0; i < 8; ++i) sim.step({});
  assert(sim.countLiveTrainingShapes() > targetShapes - 40);
  assert(sim.countLiveTrainingShapes() <= targetShapes);
}
}

std::string runScript(std::uint64_t seed, int ticks, const std::string& scenario) {
  Simulation sim(Config{seed, 3, ticks, scenario});
  for (int tick = 0; tick < ticks; ++tick) {
    std::vector<Action> actions;
    actions.push_back(Action{0, 1.0, 0.0, 1.0, 0.0, tick % 4 == 0, false, -1, -1});
    actions.push_back(Action{1, -0.5, 0.25, -1.0, 0.0, tick % 5 == 0, false, -1, -1});
    actions.push_back(Action{2, 0.0, -1.0, 0.0, -1.0, tick % 7 == 0, true, -1, -1});
    const auto result = sim.step(actions);
    assert(result.tick == tick + 1);
    assert(result.rewards.size() == 3);
  }
  return sim.fullWorldSnapshotJson();
}

int main() {
  const std::string a = runScript(123, 32, "agents-projectiles");
  const std::string b = runScript(123, 32, "agents-projectiles");
  const std::string c = runScript(124, 32, "agents-projectiles");
  assert(a == b);
  assert(a != c);

  Simulation grid(Config{123, 1, 5, "rl-grid-smoke"});
  const std::string beforeObs = grid.fullWorldSnapshotJson();
  const int combatGridCount = grid.combatGridFloatCount();
  assert(combatGridCount == CombatChannelCount * 21 * 21);
  assert(grid.writeCombatGrid(0, nullptr, 0) == combatGridCount);
  std::vector<float> combatGrid(static_cast<std::size_t>(combatGridCount));
  assert(grid.writeCombatGrid(0, combatGrid.data(), combatGridCount) == combatGridCount);
  assert(std::any_of(combatGrid.begin(), combatGrid.end(), [](float value) { return value > 0.0f; }));
  const int combatSelfCount = grid.combatSelfFloatCount();
  assert(combatSelfCount == 27);
  std::vector<float> combatSelf(static_cast<std::size_t>(combatSelfCount), -1.0f);
  assert(grid.writeCombatSelf(0, combatSelf.data(), combatSelfCount) == combatSelfCount);
  assert(combatSelf[0] == 1.0f);
  assert(combatSelf[8] > 0.0f);
  const int combatPrevCount = grid.combatPrevActionFloatCount();
  assert(combatPrevCount == 5);
  std::vector<float> combatPrev(static_cast<std::size_t>(combatPrevCount), -1.0f);
  assert(grid.writeCombatPrevAction(0, combatPrev.data(), combatPrevCount) == combatPrevCount);
  for (float value : combatPrev) assert(value == 0.0f);
  assert(grid.writeCombatGrid(9999, combatGrid.data(), combatGridCount) == -1);
  assert(grid.fullWorldSnapshotJson() == beforeObs);
  assert(grid.episodeStatsFieldCount() == EpisodeStatsFieldCount);
  std::vector<double> episodeStats(static_cast<std::size_t>(EpisodeStatsFieldCount), -1.0);
  assert(grid.writeEpisodeStats(episodeStats.data(), EpisodeStatsFieldCount) == EpisodeStatsFieldCount);
  assert(episodeStats[ShotsFiredIndex] == 0.0);

  const auto combatStep = grid.step({Action{0, 1.0, -1.0, 0.0, 1.0, true, false, -1, -1}});
  assert(combatStep.tick == 1);
  std::fill(combatPrev.begin(), combatPrev.end(), -1.0f);
  assert(grid.writeCombatPrevAction(0, combatPrev.data(), combatPrevCount) == combatPrevCount);
  assert(combatPrev[0] == 1.0f);
  assert(combatPrev[1] == -1.0f);
  assert(combatPrev[2] == 0.0f);
  assert(combatPrev[3] == 1.0f);
  assert(combatPrev[4] == 1.0f);
  std::fill(episodeStats.begin(), episodeStats.end(), -1.0);
  assert(grid.writeEpisodeStats(episodeStats.data(), EpisodeStatsFieldCount) == EpisodeStatsFieldCount);
  assert(episodeStats[ShotsFiredIndex] == 1.0);


  Simulation killStats(Config{123, 1, 5, "rl-grid-smoke"});
  Simulation::Entity source;
  source.id = 0;
  source.kind = "agent";
  source.ownerId = -1;
  Simulation::Entity farmTarget;
  farmTarget.kind = "shape";
  farmTarget.scoreReward = 25;
  killStats.recordScoreReward(source, farmTarget);
  Simulation::Entity projectileTarget;
  projectileTarget.kind = "projectile";
  projectileTarget.scoreReward = 0;
  killStats.recordScoreReward(source, projectileTarget);
  Simulation::Entity enemyTarget;
  enemyTarget.kind = "agent";
  enemyTarget.scoreReward = 100;
  killStats.recordScoreReward(source, enemyTarget);
  killStats.recordKill(source, enemyTarget);
  killStats.recordDamageDealt(source, farmTarget, 5.0);
  killStats.recordDamageDealt(source, enemyTarget, 7.0);
  std::vector<double> killStatsValues(static_cast<std::size_t>(EpisodeStatsFieldCount), -1.0);
  assert(killStats.writeEpisodeStats(killStatsValues.data(), EpisodeStatsFieldCount) == EpisodeStatsFieldCount);
  assert(killStatsValues[EnemyKillsIndex] == 1.0);
  assert(killStatsValues[FarmKillsIndex] == 1.0);
  assert(killStatsValues[DamageDealtIndex] == 12.0);
  assert(killStatsValues[EnemyDamageDealtIndex] == 7.0);

  Simulation upgrade(Config{123, 1, 8, "upgrade-ready"});
  std::vector<float> upgradeSelf(static_cast<std::size_t>(upgrade.combatSelfFloatCount()), -1.0f);
  assert(upgrade.writeCombatSelf(0, upgradeSelf.data(), upgrade.combatSelfFloatCount()) == upgrade.combatSelfFloatCount());
  assert(upgradeSelf[1] == 1.0f);
  assert(upgradeSelf[2] == 1.0f);
  assert(upgradeSelf[3] == 1.0f);
  assert(upgradeSelf[9] == 0.0f);
  const auto upgradeStep = upgrade.step({Action{0, 0.0, 0.0, 1.0, 0.0, false, false, 0, 0}});
  assert(upgradeStep.tick == 1);
  std::vector<double> upgradeStats(static_cast<std::size_t>(EpisodeStatsFieldCount), -1.0);
  assert(upgrade.writeEpisodeStats(upgradeStats.data(), EpisodeStatsFieldCount) == EpisodeStatsFieldCount);
  assert(upgradeStats[UpgradeChoicesIndex] > 0.0);

  Simulation empty(Config{123, 0, 5, "empty-arena"});
  const auto done = empty.step({});
  assert(done.done);
  assert(empty.activeEntityCount() == 0);

  Simulation denseA(Config{1, 4, 200, "dense-collision"});
  Simulation denseB(Config{1, 4, 200, "dense-collision"});
  const auto denseResult = denseA.stepMany({Action{0, 0.0, 0.0, 1.0, 0.0, false, false, -1, -1},
                                            Action{1, 0.0, 0.0, 1.0, 0.0, false, false, -1, -1},
                                            Action{2, 0.0, 0.0, 1.0, 0.0, false, false, -1, -1},
                                            Action{3, 0.0, 0.0, 1.0, 0.0, false, false, -1, -1}},
                                           10);
  assert(denseResult.tick >= 6 && denseResult.tick <= 10);
  std::vector<double> denseAStats(static_cast<std::size_t>(4 * EpisodeStatsFieldCount), -1.0);
  std::vector<double> denseBStats(static_cast<std::size_t>(4 * EpisodeStatsFieldCount), -1.0);
  assert(denseA.writeEpisodeStats(denseAStats.data(), static_cast<int>(denseAStats.size())) == static_cast<int>(denseAStats.size()));
  assert(denseB.writeEpisodeStats(denseBStats.data(), static_cast<int>(denseBStats.size())) == static_cast<int>(denseBStats.size()));
  double deathSumA = 0.0;
  double deathSumB = 0.0;
  for (int i = 0; i < 4; ++i) {
    deathSumA += denseAStats[static_cast<std::size_t>(i * EpisodeStatsFieldCount + DeathCountIndex)];
    deathSumB += denseBStats[static_cast<std::size_t>(i * EpisodeStatsFieldCount + DeathCountIndex)];
  }
  assert(deathSumA == 4.0);
  assert(deathSumB == 0.0);

  assertTrainingScenario("training-ffa-easy", 3200.0, 60);
  assertTrainingScenario("training-ffa-medium", 4800.0, 100);
  assertTrainingScenario("training-ffa-hard", 6400.0, 150);

  grid.reset(123);
  std::fill(episodeStats.begin(), episodeStats.end(), -1.0);
  assert(grid.writeEpisodeStats(episodeStats.data(), EpisodeStatsFieldCount) == EpisodeStatsFieldCount);
  assert(episodeStats[ShotsFiredIndex] == 0.0);

  std::cout << "headless deterministic smoke passed\n";
  return 0;
}
