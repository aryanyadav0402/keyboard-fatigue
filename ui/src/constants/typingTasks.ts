export type TypingTask = {
  id: number
  type: string
  instruction: string
  source: string
  prompt: string
  expected: string
}

export const QUESTION_POOL_ID = 'transform-v1'

export const TYPING_TASKS: TypingTask[] = [
  {
    id: 1,
    type: 'capitalize_vowels',
    instruction: 'Type this sentence, but CAPITALIZE all vowels (a, e, i, o, u):',
    source: 'the calico cat gave a pretty flower to the angry cow',
    prompt: "Type this sentence, but CAPITALIZE all vowels (a, e, i, o, u): 'the calico cat gave a pretty flower to the angry cow'",
    expected: 'thE cAlIcO cAt gAvE A prEtty flOwEr tO thE Angry cOw',
  },
  {
    id: 2,
    type: 'dashes_for_spaces',
    instruction: 'Replace every space with a dash (-):',
    source: 'a sleepy panda shared a sweet red apple with a tiny duck',
    prompt: "Replace every space with a dash (-): 'a sleepy panda shared a sweet red apple with a tiny duck'",
    expected: 'a-sleepy-panda-shared-a-sweet-red-apple-with-a-tiny-duck',
  },
  {
    id: 3,
    type: 'dots_and_vowels',
    instruction: 'Replace every space with a dot (.) and CAPITALIZE all vowels:',
    source: 'the baby otter floated down the calm blue river',
    prompt: "Replace every space with a dot (.) and CAPITALIZE all vowels: 'the baby otter floated down the calm blue river'",
    expected: 'thE.bAby.OttEr.flOAtEd.dOwn.thE.cAlm.blUE.rIvEr',
  },
  {
    id: 4,
    type: 'dashes_and_vowels',
    instruction: 'Replace spaces with dashes (-) and CAPITALIZE all vowels:',
    source: 'the polite pig invited the hedgehog to tea',
    prompt: "Replace spaces with dashes (-) and CAPITALIZE all vowels: 'the polite pig invited the hedgehog to tea'",
    expected: 'thE-pOlItE-pIg-InvItEd-thE-hEdgEhOg-tO-tEA',
  },
  {
    id: 5,
    type: 'capitalize_first_and_last',
    instruction: 'Capitalize the FIRST and LAST letter of each word:',
    source: 'the clumsy bear dropped his favorite jar of honey',
    prompt: "Capitalize the FIRST and LAST letter of each word: 'the clumsy bear dropped his favorite jar of honey'",
    expected: 'ThE ClumsY BeaR DroppeD HiS FavoritE JaR OF HoneY',
  },
  {
    id: 6,
    type: 'capitalize_vowels',
    instruction: 'Type this sentence, but CAPITALIZE all vowels (a, e, i, o, u):',
    source: 'a curious raccoon found a shiny golden key',
    prompt: "Type this sentence, but CAPITALIZE all vowels (a, e, i, o, u): 'a curious raccoon found a shiny golden key'",
    expected: 'A cUrIOUs rAccOOn fOUnd A shIny gOldEn kEy',
  },
]

export function selectTypingTask(): TypingTask {
  return TYPING_TASKS[Math.floor(Math.random() * TYPING_TASKS.length)]
}
